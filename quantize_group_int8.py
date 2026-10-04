"""Create and verify six group-only INT8 QDQ models for M7.

The selected node names come from the M3 graph mapping. Each run starts from
the same FP32 ONNX graph and uses the M5 calibration reader and PTQ settings.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

from data import SPLIT_PATH, make_data_loader
from graph_mapping import GROUPS, map_graph, validate_mapping
from quantize_full_int8 import CIFAR10CalibrationReader, FP32_PATH, MAPPING_PATH, sha256


ROOT = Path(__file__).resolve().parent
GROUP_SLUGS = {
    "Initial Conv": "initial_conv",
    "Layer 1": "layer1",
    "Layer 2": "layer2",
    "Layer 3": "layer3",
    "Layer 4": "layer4",
    "FC": "fc",
}
MODEL_PATHS = {
    group: ROOT / "checkpoints" / f"group_{slug}_int8.onnx"
    for group, slug in GROUP_SLUGS.items()
}
MANIFEST_PATH = ROOT / "checkpoints" / "m7_group_manifest.json"


def load_mapping():
    """Return M3 rows, FP32 model, and hash after checking the entire mapping."""
    if not FP32_PATH.is_file():
        raise FileNotFoundError(FP32_PATH)
    if not MAPPING_PATH.is_file():
        raise FileNotFoundError(MAPPING_PATH)

    fp32_hash = sha256(FP32_PATH)
    fp32_model = onnx.load(FP32_PATH)
    onnx.checker.check_model(fp32_model)
    expected_rows = map_graph(fp32_model.graph)
    validate_mapping(expected_rows)

    with MAPPING_PATH.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    expected_fields = set(expected_rows[0]) | {"onnx_sha256"}
    if len(fields) != len(expected_fields) or set(fields) != expected_fields:
        raise ValueError("M3 매핑표의 열이 현재 그래프 매핑과 다릅니다.")
    if len(rows) != len(expected_rows):
        raise ValueError("M3 매핑표의 노드 개수가 현재 FP32 그래프와 다릅니다.")
    if {row["onnx_sha256"] for row in rows} != {fp32_hash}:
        raise ValueError("M3 매핑표의 FP32 ONNX SHA-256이 현재 모델과 다릅니다.")
    for index, (actual, expected) in enumerate(zip(rows, expected_rows)):
        if any(actual[field] != value for field, value in expected.items()):
            raise ValueError(f"M3 매핑표 {index + 1}행이 FP32 그래프와 다릅니다.")
    validate_mapping(rows)
    if tuple(GROUP_SLUGS) != GROUPS:
        raise ValueError("M7 그룹 순서가 M3 그룹 순서와 다릅니다.")
    if any(not any(row["group"] == group for row in rows) for group in GROUPS):
        raise ValueError("비어 있는 M7 그룹이 있습니다.")
    if any(row["op_type"] not in {"Conv", "Gemm", "Add"} for row in rows):
        raise ValueError("M5 양자화 대상 연산 외의 M3 노드가 있습니다.")
    return rows, fp32_model, fp32_hash


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_qparams(scale_name, zero_point_name, initializers, zero_point_dtype, context):
    scale_tensor = initializers.get(scale_name)
    zero_point_tensor = initializers.get(zero_point_name)
    _require(scale_tensor is not None and zero_point_tensor is not None,
             f"{context}: scale/zero-point initializer가 없습니다.")
    _require(scale_tensor.data_type == onnx.TensorProto.FLOAT,
             f"{context}: scale이 FP32가 아닙니다.")
    _require(zero_point_tensor.data_type == zero_point_dtype,
             f"{context}: zero-point dtype이 다릅니다.")
    scale = numpy_helper.to_array(scale_tensor)
    zero_point = numpy_helper.to_array(zero_point_tensor)
    if zero_point_dtype == onnx.TensorProto.INT32:
        # ORT stores bias scale as a one-element vector, while bias zero-point
        # is scalar. Both broadcast to the whole INT32 bias tensor.
        _require(scale.size == 1 and zero_point.size == 1
                 and scale.ndim <= 1 and zero_point.ndim <= 1,
                 f"{context}: bias scale/zero-point shape가 다릅니다.")
    else:
        _require(scale.ndim == 0 and zero_point.ndim == 0,
                 f"{context}: per-tensor scale/zero-point가 아닙니다.")
    _require(np.isfinite(scale).all() and bool(scale > 0),
             f"{context}: scale이 유한한 양수가 아닙니다.")


def _check_activation_dq(input_name, producers, initializers, context):
    dq = producers.get(input_name)
    _require(dq is not None and dq.op_type == "DequantizeLinear" and len(dq.input) == 3,
             f"{context}: activation DQ가 없습니다.")
    q = producers.get(dq.input[0])
    _require(q is not None and q.op_type == "QuantizeLinear" and len(q.input) == 3,
             f"{context}: activation Q가 DQ 앞에 없습니다.")
    _require(list(q.input[1:]) == list(dq.input[1:]),
             f"{context}: activation Q/DQ 파라미터가 다릅니다.")
    _check_qparams(dq.input[1], dq.input[2], initializers, onnx.TensorProto.INT8,
                   f"{context} activation")


def _check_output_q(node, consumers, initializers):
    _require(len(node.output) == 1, f"{node.name}: 출력 수가 예상과 다릅니다.")
    q_nodes = [consumer for consumer in consumers[node.output[0]]
               if consumer.op_type == "QuantizeLinear" and consumer.input[0] == node.output[0]]
    _require(bool(q_nodes), f"{node.name}: 출력 Q가 없습니다.")
    for q in q_nodes:
        _require(len(q.input) == 3, f"{node.name}: 출력 Q 입력 수가 다릅니다.")
        _check_qparams(q.input[1], q.input[2], initializers, onnx.TensorProto.INT8,
                       f"{node.name} 출력")


def _check_quantized_parameter(node, input_index, original_name, dtype,
                               producers, initializers, fp32_initializers):
    context = f"{node.name} {original_name}"
    _require(len(node.input) > input_index, f"{context}: 파라미터 입력이 없습니다.")
    dq = producers.get(node.input[input_index])
    _require(dq is not None and dq.op_type == "DequantizeLinear" and len(dq.input) == 3,
             f"{context}: 파라미터 DQ가 없습니다.")
    _require(dq.input[0] == f"{original_name}_quantized",
             f"{context}: 원래 파라미터에 대응하는 quantized initializer가 아닙니다.")
    quantized = initializers.get(dq.input[0])
    original = fp32_initializers.get(original_name)
    _require(quantized is not None and original is not None,
             f"{context}: initializer가 없습니다.")
    _require(quantized.data_type == dtype and list(quantized.dims) == list(original.dims),
             f"{context}: quantized dtype 또는 shape가 다릅니다.")
    _check_qparams(dq.input[1], dq.input[2], initializers, dtype, context)


def _check_original_parameter(node, input_index, original_name,
                              initializers, fp32_initializers):
    context = f"{node.name} {original_name}"
    _require(len(node.input) > input_index and node.input[input_index] == original_name,
             f"{context}: 비선택 파라미터 입력이 바뀌었습니다.")
    actual = initializers.get(original_name)
    original = fp32_initializers.get(original_name)
    _require(actual is not None and original is not None,
             f"{context}: FP32 initializer가 없습니다.")
    _require(actual.data_type == onnx.TensorProto.FLOAT
             and actual.SerializeToString() == original.SerializeToString(),
             f"{context}: 원래 FP32 initializer 값이 바뀌었습니다.")


def verify_group_model(model_path, group, rows, fp32_model):
    """Fail unless exactly this group's mapped nodes have the expected QDQ form."""
    if group not in GROUP_SLUGS:
        raise ValueError(f"알 수 없는 그룹: {group}")
    validate_mapping(rows)
    target_rows = [row for row in rows if row["group"] == group]
    _require(bool(target_rows), f"{group}: 양자화 대상 노드가 없습니다.")
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    fp32_nodes = {node.name: node for node in fp32_model.graph.node}
    nodes = {node.name: node for node in model.graph.node}
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
    fp32_initializers = {tensor.name: tensor for tensor in fp32_model.graph.initializer}
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}

    for row in rows:
        name = row["node_name"]
        reference = fp32_nodes.get(name)
        actual = nodes.get(name)
        _require(reference is not None and actual is not None
                 and reference.op_type == row["op_type"] == actual.op_type,
                 f"{name}: M3 노드 이름/연산이 보존되지 않았습니다.")
        _require([attr.SerializeToString() for attr in actual.attribute]
                 == [attr.SerializeToString() for attr in reference.attribute],
                 f"{name}: 연산 속성이 바뀌었습니다.")

    target_counts = Counter()
    nonselected_fp32 = 0
    boundary_qdq_nodes = []
    for row in rows:
        node = nodes[row["node_name"]]
        selected = row["group"] == group
        if selected:
            target_counts[node.op_type] += 1
            if node.op_type in {"Conv", "Gemm"}:
                _require(len(node.input) == 3, f"{node.name}: 입력 수가 다릅니다.")
                _check_activation_dq(node.input[0], producers, initializers, node.name)
                _check_quantized_parameter(node, 1, row["weight_initializer"],
                                           onnx.TensorProto.INT8, producers,
                                           initializers, fp32_initializers)
                bias_name = row["initializer_inputs"].split(";")[1]
                _check_quantized_parameter(node, 2, bias_name,
                                           onnx.TensorProto.INT32, producers,
                                           initializers, fp32_initializers)
            else:
                _require(len(node.input) == 2, f"{node.name}: Add 입력 수가 다릅니다.")
                for input_name in node.input:
                    _check_activation_dq(input_name, producers, initializers, node.name)
            _check_output_q(node, consumers, initializers)
        else:
            if node.op_type in {"Conv", "Gemm"}:
                _require(len(node.input) == 3, f"{node.name}: 입력 수가 다릅니다.")
                _check_original_parameter(node, 1, row["weight_initializer"],
                                          initializers, fp32_initializers)
                bias_name = row["initializer_inputs"].split(";")[1]
                _check_original_parameter(node, 2, bias_name,
                                          initializers, fp32_initializers)
                nonselected_fp32 += 1
            elif node.op_type == "Add":
                # The main path comes from this block's unselected Conv2.
                # Its direct FP32 edge distinguishes this Add from a targeted
                # Add even when Q/DQ appears on the skip/output boundaries.
                _require(node.input[0] == fp32_nodes[node.name].input[0],
                         f"{node.name}: 비선택 Add의 FP32 메인 경로가 바뀌었습니다.")
            # A selected neighbor may put Q/DQ on this FP32 node's activation.
            if (any(producers.get(name) is not None
                    and producers[name].op_type == "DequantizeLinear" for name in node.input)
                    or any(consumer.op_type == "QuantizeLinear" for output in node.output
                           for consumer in consumers[output])):
                boundary_qdq_nodes.append(node.name)

    _require(sum(target_counts.values()) == len(target_rows),
             f"{group}: 검증한 대상 노드 수가 다릅니다.")
    return {
        "mapped_nodes_preserved": len(rows),
        "selected_node_counts": dict(target_counts),
        "nonselected_fp32_conv_gemm": nonselected_fp32,
        "boundary_qdq_nonselected_nodes": boundary_qdq_nodes,
        "qdq_node_count": sum(node.op_type in {"QuantizeLinear", "DequantizeLinear"}
                              for node in model.graph.node),
    }


def quantize_group_models():
    rows, fp32_model, fp32_hash = load_mapping()
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"기존 M5 split 파일이 없습니다: {SPLIT_PATH}")
    split_hash = sha256(SPLIT_PATH)
    _, _, cal_loader, _ = make_data_loader()
    if len(cal_loader.dataset) != 1000:
        raise ValueError("Calibration 데이터는 정확히 1,000장이어야 합니다.")
    if sha256(SPLIT_PATH) != split_hash:
        raise ValueError("Calibration split 파일이 로딩 중 바뀌었습니다.")
    _require(len(fp32_model.graph.input) == 1, "FP32 ONNX 입력 수를 확인하세요.")
    input_name = fp32_model.graph.input[0].name

    manifest = {
        "fp32_onnx_sha256": fp32_hash,
        "mapping_sha256": sha256(MAPPING_PATH),
        "split_sha256": split_hash,
        "calibration_samples": 1000,
        "settings": {
            "quant_format": "QDQ",
            "calibrate_method": "MinMax",
            "activation_type": "QInt8",
            "weight_type": "QInt8",
            "op_types_to_quantize": ["Conv", "Gemm", "Add"],
            "per_channel": False,
            "reduce_range": False,
            "calibration_providers": ["CPUExecutionProvider"],
        },
        "software": {"onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__},
        "models": [],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        path = MODEL_PATHS[group]
        temporary_path = path.with_name(f"{path.stem}.tmp.onnx")
        target_nodes = [row["node_name"] for row in rows if row["group"] == group]
        _require(bool(target_nodes), f"{group}: 양자화 대상이 비어 있습니다.")
        reader = CIFAR10CalibrationReader(cal_loader, input_name)
        try:
            # Match M5 exactly, adding only the M3-derived node selection.
            quantize_static(
                model_input=FP32_PATH,
                model_output=temporary_path,
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                calibrate_method=CalibrationMethod.MinMax,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["Conv", "Gemm", "Add"],
                nodes_to_quantize=target_nodes,
                per_channel=False,
                reduce_range=False,
                calibration_providers=["CPUExecutionProvider"],
            )
            report = verify_group_model(temporary_path, group, rows, fp32_model)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
        manifest["models"].append({
            "group": group,
            "file": path.name,
            "onnx_sha256": sha256(path),
            "target_nodes": target_nodes,
            "graph_verification": report,
        })
        print(f"{group}: {len(target_nodes)}개 대상 노드 검증 완료 → {path}")

    _require(sha256(SPLIT_PATH) == split_hash,
             "Calibration split 파일이 양자화 중 바뀌었습니다.")
    temporary_manifest = MANIFEST_PATH.with_name(f"{MANIFEST_PATH.stem}.tmp.json")
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary_manifest.replace(MANIFEST_PATH)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    print(f"M7 manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    quantize_group_models()
