"""Create and check the three M8 selective INT8 candidates.

Every candidate starts from the same M3 FP32 ONNX model. Groups listed in a
candidate stay FP32; the other mapped Conv/Gemm/Add nodes use M5's static PTQ
settings and the same 1,000-image calibration split.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import onnx
import onnxruntime
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

from data import SPLIT_PATH, make_data_loader
from graph_mapping import GROUPS, validate_mapping
from quantize_full_int8 import CIFAR10CalibrationReader, FP32_PATH, MAPPING_PATH, sha256
from quantize_group_int8 import (
    MANIFEST_PATH as M7_MANIFEST_PATH,
    _check_activation_dq,
    _check_original_parameter,
    _check_output_q,
    _check_quantized_parameter,
    _require,
    load_mapping,
)
from export_onnx import load_onnx


ROOT = Path(__file__).resolve().parent
CANDIDATES = {
    "keep_layer1": ("Layer 1",),
    "keep_layer3": ("Layer 3",),
    "keep_layer1_layer3": ("Layer 1", "Layer 3"),
}
MODEL_PATHS = {
    name: ROOT / "checkpoints" / f"selective_{name}_int8.onnx"
    for name in CANDIDATES
}
MANIFEST_PATH = ROOT / "checkpoints" / "m8_selective_manifest.json"
PTQ_SETTINGS = {
    "quant_format": "QDQ",
    "calibrate_method": "MinMax",
    "activation_type": "QInt8",
    "weight_type": "QInt8",
    "op_types_to_quantize": ["Conv", "Gemm", "Add"],
    "per_channel": False,
    "reduce_range": False,
    "calibration_providers": ["CPUExecutionProvider"],
}


def verify_selective_model(model_path, keep_groups, rows, fp32_model):
    """Check that all and only the intended mapped nodes have INT8 QDQ."""
    validate_mapping(rows)
    keep_groups = tuple(keep_groups)
    _require(len(keep_groups) == len(set(keep_groups))
             and set(keep_groups) <= set(GROUPS),
             "FP32 유지 그룹이 중복되거나 M3 그룹과 다릅니다.")
    _require(set(keep_groups) != set(GROUPS), "양자화할 그룹이 없습니다.")
    target_rows = [row for row in rows if row["group"] not in keep_groups]
    _require(bool(target_rows), "양자화할 M3 노드가 없습니다.")

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    fp32_nodes = {node.name: node for node in fp32_model.graph.node}
    nodes = {node.name: node for node in model.graph.node}
    _require(len(nodes) == len(model.graph.node), "ONNX 그래프에 중복된 노드 이름이 있습니다.")
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    fp32_initializers = {tensor.name: tensor for tensor in fp32_model.graph.initializer}

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

    quantized_counts = Counter()
    kept_counts = Counter()
    boundary_qdq_kept_nodes = []
    for row in rows:
        node = nodes[row["node_name"]]
        if row["group"] not in keep_groups:
            quantized_counts[node.op_type] += 1
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
            kept_counts[node.op_type] += 1
            reference = fp32_nodes[node.name]
            if node.op_type in {"Conv", "Gemm"}:
                _require(len(node.input) == 3, f"{node.name}: 입력 수가 다릅니다.")
                _check_original_parameter(node, 1, row["weight_initializer"],
                                          initializers, fp32_initializers)
                bias_name = row["initializer_inputs"].split(";")[1]
                _check_original_parameter(node, 2, bias_name,
                                          initializers, fp32_initializers)
            elif node.op_type == "Add":
                # This main edge comes from a kept Conv in the same block.
                # Quantized neighboring groups can still add QDQ at boundaries.
                _require(node.input[0] == reference.input[0],
                         f"{node.name}: FP32 유지 Add의 메인 경로가 바뀌었습니다.")
            if (any(producers.get(name) is not None
                    and producers[name].op_type == "DequantizeLinear"
                    for name in node.input)
                    or any(consumer.op_type == "QuantizeLinear" for output in node.output
                           for consumer in consumers[output])):
                boundary_qdq_kept_nodes.append(node.name)

    _require(sum(quantized_counts.values()) == len(target_rows),
             "양자화 대상 노드 수가 M3 매핑과 다릅니다.")
    _require(sum(kept_counts.values()) == len(rows) - len(target_rows),
             "FP32 유지 노드 수가 M3 매핑과 다릅니다.")
    return {
        "mapped_nodes_preserved": len(rows),
        "quantized_node_counts": dict(quantized_counts),
        "kept_node_counts": dict(kept_counts),
        "boundary_qdq_kept_nodes": boundary_qdq_kept_nodes,
        "qdq_node_count": sum(node.op_type in {"QuantizeLinear", "DequantizeLinear"}
                              for node in model.graph.node),
    }


def quantize_selective_models():
    rows, fp32_model, fp32_hash = load_mapping()
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"기존 calibration split 파일이 없습니다: {SPLIT_PATH}")
    split_hash = sha256(SPLIT_PATH)
    if not M7_MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"M7 생성 기록이 없습니다: {M7_MANIFEST_PATH}")
    m7_manifest = json.loads(M7_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (m7_manifest["fp32_onnx_sha256"] != fp32_hash
            or m7_manifest["mapping_sha256"] != sha256(MAPPING_PATH)
            or m7_manifest["split_sha256"] != split_hash
            or m7_manifest["calibration_samples"] != 1000
            or m7_manifest["settings"] != PTQ_SETTINGS):
        raise ValueError("M7과 FP32 모델·매핑·split·양자화 설정이 다릅니다.")
    _, _, cal_loader, _ = make_data_loader()
    _require(len(cal_loader.dataset) == 1000, "Calibration 데이터는 1,000장이어야 합니다.")
    _require(sha256(SPLIT_PATH) == split_hash,
             "Calibration split 파일이 로딩 중 바뀌었습니다.")
    _require(len(fp32_model.graph.input) == 1, "FP32 ONNX 입력 수를 확인하세요.")
    input_name = fp32_model.graph.input[0].name

    manifest = {
        "fp32_onnx_sha256": fp32_hash,
        "m7_manifest_sha256": sha256(M7_MANIFEST_PATH),
        "mapping_sha256": sha256(MAPPING_PATH),
        "split_sha256": split_hash,
        "calibration_samples": 1000,
        "settings": PTQ_SETTINGS,
        "software": {"onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__},
        "candidates": [],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_images = next(iter(cal_loader))[0]
    for candidate, keep_groups in CANDIDATES.items():
        path = MODEL_PATHS[candidate]
        temporary_path = path.with_name(f"{path.stem}.tmp.onnx")
        target_nodes = [row["node_name"] for row in rows if row["group"] not in keep_groups]
        kept_nodes = [row["node_name"] for row in rows if row["group"] in keep_groups]
        _require(len(target_nodes) + len(kept_nodes) == len(rows),
                 f"{candidate}: 대상/유지 노드 수가 다릅니다.")
        _require(bool(target_nodes), f"{candidate}: 양자화 대상이 비어 있습니다.")
        reader = CIFAR10CalibrationReader(cal_loader, input_name)
        try:
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
            graph_report = verify_selective_model(temporary_path, keep_groups,
                                                  rows, fp32_model)
            session = load_onnx(temporary_path)
            for batch_size in (1, 32):
                output = session.run(None, {
                    input_name: sample_images[:batch_size].numpy()
                })[0]
                _require(output.shape == (batch_size, 10),
                         f"{candidate}: batch {batch_size} 출력 shape가 다릅니다.")
            del session
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
        manifest["candidates"].append({
            "candidate": candidate,
            "file": path.name,
            "onnx_sha256": sha256(path),
            "keep_groups": list(keep_groups),
            "kept_nodes": kept_nodes,
            "target_nodes": target_nodes,
            "graph_verification": graph_report,
        })
        print(f"{candidate}: FP32 {len(kept_nodes)}개, INT8 {len(target_nodes)}개 검증 완료")

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
    print(f"M8 manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    quantize_selective_models()
