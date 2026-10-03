"""Map the exported ResNet-18 ONNX quantization targets to analysis groups.

This map is for the current FP32 ONNX graph. If the graph changes, rerun this
script and review the result before selecting nodes for quantization.
"""

import csv
import hashlib
from collections import Counter
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT / "checkpoints" / "onnx_resnet18.onnx"
OUTPUT_PATH = ROOT / "onnx_graph_mapping.csv"
GROUPS = ("Initial Conv", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "FC")


def group_from_weight(weight_name):
    if weight_name == "conv1.weight":
        return "Initial Conv"
    if weight_name == "fc.weight":
        return "FC"

    stage = weight_name.split(".", maxsplit=1)[0]
    if stage in {"layer1", "layer2", "layer3", "layer4"} and weight_name.endswith(
        ".weight"
    ):
        return f"Layer {stage[-1]}"

    raise ValueError(f"분석 그룹을 알 수 없는 weight: {weight_name}")


def map_graph(graph):
    initializers = {tensor.name for tensor in graph.initializer}
    producers = {output: node for node in graph.node for output in node.output}
    rows = []

    for node in graph.node:
        if node.op_type in {"Conv", "Gemm", "MatMul"}:
            if len(node.input) < 2 or node.input[1] not in initializers:
                raise ValueError(f"weight initializer를 찾을 수 없음: {node.name}")
            weight_name = node.input[1]
            group = group_from_weight(weight_name)
            initializer_inputs = [name for name in node.input[1:] if name in initializers]
            main_path_weight = ""

        elif node.op_type == "Add":
            # 이 ResNet-18 export에서는 Add의 첫 입력이 block의 메인 Conv 경로다.
            main_conv = producers.get(node.input[0])
            if main_conv is None or main_conv.op_type != "Conv":
                raise ValueError(f"Add의 메인 Conv를 확인할 수 없음: {node.name}")
            main_path_weight = main_conv.input[1]
            group = group_from_weight(main_path_weight)
            if group not in GROUPS[1:5]:
                raise ValueError(f"예상 밖의 Add 그룹: {node.name}, {group}")
            weight_name = ""
            initializer_inputs = []

        else:
            continue

        rows.append(
            {
                "group": group,
                "op_type": node.op_type,
                "node_name": node.name,
                "weight_initializer": weight_name,
                "initializer_inputs": ";".join(initializer_inputs),
                "main_path_weight_for_add": main_path_weight,
                "inputs": ";".join(node.input),
                "outputs": ";".join(node.output),
            }
        )

    return rows


def validate_mapping(rows):
    counts = Counter((row["group"], row["op_type"]) for row in rows)
    expected = {
        ("Initial Conv", "Conv"): 1,
        ("Layer 1", "Conv"): 4,
        ("Layer 1", "Add"): 2,
        ("Layer 2", "Conv"): 5,
        ("Layer 2", "Add"): 2,
        ("Layer 3", "Conv"): 5,
        ("Layer 3", "Add"): 2,
        ("Layer 4", "Conv"): 5,
        ("Layer 4", "Add"): 2,
    }
    for key, count in expected.items():
        if counts[key] != count:
            raise ValueError(f"{key} 개수 불일치: 예상 {count}, 실제 {counts[key]}")

    fc_rows = [row for row in rows if row["group"] == "FC"]
    if len(fc_rows) != 1 or fc_rows[0]["op_type"] not in {"Gemm", "MatMul"}:
        raise ValueError(f"FC 연산 확인 필요: {fc_rows}")

    if len(rows) != 29 or len({row["node_name"] for row in rows}) != len(rows):
        raise ValueError("대상 노드 개수 또는 이름 중복 확인 필요")

    for group in ("Layer 2", "Layer 3", "Layer 4"):
        downsample = [
            row for row in rows
            if row["group"] == group and ".downsample." in row["weight_initializer"]
        ]
        if len(downsample) != 1:
            raise ValueError(f"{group} downsample Conv 확인 필요: {downsample}")


def main():
    graph = onnx.load(ONNX_PATH).graph
    rows = map_graph(graph)
    validate_mapping(rows)

    graph_sha256 = hashlib.sha256(ONNX_PATH.read_bytes()).hexdigest()
    for row in rows:
        row["onnx_sha256"] = graph_sha256

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["group"] for row in rows)
    for group in GROUPS:
        print(f"{group}: {counts[group]}개")
    print(f"ONNX SHA256: {graph_sha256}")
    print(f"매핑표: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
