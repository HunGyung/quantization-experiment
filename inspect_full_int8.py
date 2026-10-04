import csv
from collections import Counter, defaultdict

import onnx

from data import make_data_loader
from evaluation import evaluate_validation
from export_onnx import load_onnx
from quantize_full_int8 import (
    FP32_PATH,
    INT8_PATH,
    MAPPING_PATH,
    check_fp32_model,
)


def inspect_quantized_graph():
    check_fp32_model()
    with MAPPING_PATH.open(newline="", encoding="utf-8-sig") as file:
        mapped_nodes = list(csv.DictReader(file))

    model = onnx.load(INT8_PATH)
    onnx.checker.check_model(model)
    nodes = {node.name: node for node in model.graph.node}
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}

    counts = Counter()
    missing = []
    for row in mapped_nodes:
        node = nodes.get(row["node_name"])
        if node is None or node.op_type != row["op_type"]:
            missing.append(row["node_name"])
            continue

        if node.op_type in {"Conv", "Gemm"}:
            activation_dq = producers.get(node.input[0])
            weight_dq = producers.get(node.input[1])
            if weight_dq is None or weight_dq.op_type != "DequantizeLinear":
                missing.append(node.name)
                continue
            weight = initializers.get(weight_dq.input[0])
            has_params = len(weight_dq.input) >= 3 and all(
                name in initializers for name in weight_dq.input[1:3]
            )
            bias_dq = producers.get(node.input[2]) if len(node.input) > 2 else None
            bias = initializers.get(bias_dq.input[0]) if bias_dq else None
            output_has_q = any(
                consumer.op_type == "QuantizeLinear"
                for consumer in consumers[node.output[0]]
            )
            if (
                activation_dq is None
                or activation_dq.op_type != "DequantizeLinear"
                or weight is None
                or weight.data_type != onnx.TensorProto.INT8
                or not has_params
                or bias_dq is None
                or bias_dq.op_type != "DequantizeLinear"
                or bias is None
                or bias.data_type != onnx.TensorProto.INT32
                or not output_has_q
            ):
                missing.append(node.name)
                continue
            counts[node.op_type] += 1

        elif node.op_type == "Add":
            inputs_have_dq = all(
                producers.get(name) is not None
                and producers[name].op_type == "DequantizeLinear"
                for name in node.input
            )
            output_has_q = any(
                consumer.op_type == "QuantizeLinear"
                for consumer in consumers[node.output[0]]
            )
            if not inputs_have_dq or not output_has_q:
                missing.append(node.name)
                continue
            counts["Add"] += 1

    print("양자화 범위 (M3 매핑 기준):")
    print(f"  INT8 weight·INT32 bias·입출력 Q/DQ: Conv {counts['Conv']}/20, Gemm {counts['Gemm']}/1")
    print(f"  입력 DQ + 출력 Q 확인: Add {counts['Add']}/8")
    print(f"  확인되지 않은 노드: {missing if missing else '없음'}")
    excluded_ops = Counter(
        node.op_type for node in model.graph.node
        if node.op_type not in {"Conv", "Gemm", "Add", "QuantizeLinear", "DequantizeLinear"}
    )
    print(f"  이번 양자화 대상에서 제외한 원래 연산: {dict(excluded_ops)}")
    print("  Q/DQ 패턴은 그래프상의 양자화 표현이며 실제 INT8 커널 사용 증거는 아닙니다.")


def compare_validation_accuracy():
    _, val_loader, _, _ = make_data_loader()
    fp32 = evaluate_validation(load_onnx(FP32_PATH), val_loader)
    int8 = evaluate_validation(load_onnx(INT8_PATH), val_loader)

    print("Validation 5,000장 (중간 확인):")
    print(f"  FP32: loss={fp32['loss']:.4f}, accuracy={fp32['accuracy_percent']:.2f}%")
    print(f"  Full INT8: loss={int8['loss']:.4f}, accuracy={int8['accuracy_percent']:.2f}%")
    print(
        "  accuracy drop: "
        f"{fp32['accuracy_percent'] - int8['accuracy_percent']:.2f} %p"
    )


if __name__ == "__main__":
    inspect_quantized_graph()
    compare_validation_accuracy()
