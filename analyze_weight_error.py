"""Compare FP32 ONNX weights with the weights stored in a QDQ INT8 model."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parent
FP32_PATH = ROOT / "checkpoints" / "onnx_resnet18.onnx"
INT8_PATH = ROOT / "checkpoints" / "full_int8.onnx"
MAPPING_PATH = ROOT / "onnx_graph_mapping.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "generated" / "m6"
GROUPS = ("Initial Conv", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "FC")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dequantize_weight(quantized, scale, zero_point, axis):
    """Apply the DequantizeLinear parameters to an INT8 weight tensor."""
    scale = np.asarray(scale, dtype=np.float64)
    zero_point = np.asarray(zero_point, dtype=np.float64)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Weight scale은 양의 유한한 값이어야 합니다.")

    if scale.ndim == 0:
        if zero_point.ndim != 0:
            raise ValueError("Per-tensor zero-point shape가 scale과 다릅니다.")
        return (quantized.astype(np.float64) - zero_point) * scale, None

    if scale.ndim != 1:
        raise NotImplementedError("현재는 per-tensor와 per-axis weight만 지원합니다.")
    if zero_point.ndim not in (0, 1) or (
        zero_point.ndim == 1 and zero_point.shape != scale.shape
    ):
        raise ValueError("Per-axis zero-point shape가 scale과 다릅니다.")

    if not -quantized.ndim <= axis < quantized.ndim:
        raise ValueError("Per-axis axis가 weight 차원 범위를 벗어났습니다.")
    axis = axis % quantized.ndim
    if scale.size != quantized.shape[axis]:
        raise ValueError("Per-axis scale 길이가 weight의 해당 축과 다릅니다.")
    broadcast_shape = [1] * quantized.ndim
    broadcast_shape[axis] = scale.size
    scale = scale.reshape(broadcast_shape)
    if zero_point.ndim == 1:
        zero_point = zero_point.reshape(broadcast_shape)
    return (quantized.astype(np.float64) - zero_point) * scale, axis


def load_mapping(fp32_sha256):
    with MAPPING_PATH.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if {row["onnx_sha256"] for row in rows} != {fp32_sha256}:
        raise ValueError("매핑표의 FP32 ONNX SHA-256이 현재 모델과 다릅니다.")

    weight_rows = [row for row in rows if row["op_type"] in {"Conv", "Gemm"}]
    if len(weight_rows) != 21 or len({row["node_name"] for row in weight_rows}) != 21:
        raise ValueError("Conv/Gemm weight 매핑 21개를 확인할 수 없습니다.")
    if {row["group"] for row in weight_rows} != set(GROUPS):
        raise ValueError("여섯 분석 그룹이 모두 포함되지 않았습니다.")
    return weight_rows


def extract_weight(row, fp32_nodes, int8_nodes, fp32_initializers,
                   int8_initializers, int8_producers):
    name = row["node_name"]
    fp32_node = fp32_nodes.get(name)
    int8_node = int8_nodes.get(name)
    if fp32_node is None or int8_node is None:
        raise ValueError(f"대응 노드를 찾을 수 없습니다: {name}")
    if fp32_node.op_type != row["op_type"] or int8_node.op_type != row["op_type"]:
        raise ValueError(f"연산 종류가 매핑표와 다릅니다: {name}")
    if [attr.SerializeToString() for attr in fp32_node.attribute] != [
        attr.SerializeToString() for attr in int8_node.attribute
    ]:
        raise ValueError(f"연산 속성이 바뀌었습니다: {name}")

    weight_name = row["weight_initializer"]
    if fp32_node.input[1] != weight_name or weight_name not in fp32_initializers:
        raise ValueError(f"FP32 weight 매핑이 다릅니다: {name}")
    reference = numpy_helper.to_array(fp32_initializers[weight_name])
    if not np.issubdtype(reference.dtype, np.floating):
        raise ValueError(f"FP32 weight dtype을 확인하세요: {name}")

    dq = int8_producers.get(int8_node.input[1])
    if dq is None or dq.op_type != "DequantizeLinear" or len(dq.input) < 2:
        raise ValueError(f"INT8 weight의 DequantizeLinear을 찾을 수 없습니다: {name}")
    if any(attr.name == "block_size" for attr in dq.attribute):
        raise NotImplementedError(f"Blocked quantization은 지원하지 않습니다: {name}")
    if any(input_name not in int8_initializers for input_name in dq.input[:2]):
        raise ValueError(f"INT8 weight/scale initializer가 없습니다: {name}")
    if len(dq.input) > 2 and dq.input[2] and dq.input[2] not in int8_initializers:
        raise ValueError(f"INT8 zero-point initializer가 없습니다: {name}")

    quantized = numpy_helper.to_array(int8_initializers[dq.input[0]])
    scale = numpy_helper.to_array(int8_initializers[dq.input[1]])
    zero_point = (
        numpy_helper.to_array(int8_initializers[dq.input[2]])
        if len(dq.input) > 2 and dq.input[2]
        else np.array(0, dtype=quantized.dtype)
    )
    if quantized.dtype != np.int8 or zero_point.dtype != np.int8:
        raise ValueError(f"INT8 weight/zero-point dtype이 아닙니다: {name}")
    if reference.shape != quantized.shape:
        raise ValueError(f"FP32/INT8 weight shape가 다릅니다: {name}")

    axis = next((attr.i for attr in dq.attribute if attr.name == "axis"), 1)
    reconstructed, used_axis = dequantize_weight(quantized, scale, zero_point, axis)
    return reference, reconstructed, quantized, scale, zero_point, used_axis, dq


def calculate_errors(weight_rows, fp32_model, int8_model):
    fp32_nodes = {node.name: node for node in fp32_model.graph.node}
    int8_nodes = {node.name: node for node in int8_model.graph.node}
    fp32_initializers = {item.name: item for item in fp32_model.graph.initializer}
    int8_initializers = {item.name: item for item in int8_model.graph.initializer}
    int8_producers = {
        output: node for node in int8_model.graph.node for output in node.output
    }

    per_weight = []
    group_totals = {group: {"sse": 0.0, "energy": 0.0, "elements": 0, "weights": 0}
                    for group in GROUPS}
    plot_values = defaultdict(list)

    for row in weight_rows:
        reference, reconstructed, quantized, scale, zero_point, axis, dq = extract_weight(
            row, fp32_nodes, int8_nodes, fp32_initializers,
            int8_initializers, int8_producers
        )
        difference = reference.astype(np.float64) - reconstructed
        sse = float(np.sum(difference ** 2, dtype=np.float64))
        energy = float(np.sum(reference.astype(np.float64) ** 2, dtype=np.float64))
        if energy == 0:
            raise ValueError(f"FP32 weight 제곱합이 0입니다: {row['weight_initializer']}")

        group = row["group"]
        group_totals[group]["sse"] += sse
        group_totals[group]["energy"] += energy
        group_totals[group]["elements"] += reference.size
        group_totals[group]["weights"] += 1
        plot_values[group].append((reference.ravel(), reconstructed.astype(np.float32).ravel()))

        per_weight.append({
            "group": group,
            "op_type": row["op_type"],
            "node_name": row["node_name"],
            "fp32_weight": row["weight_initializer"],
            "int8_weight": dq.input[0],
            "shape": json.dumps(list(reference.shape)),
            "elements": reference.size,
            "quantization": "per-tensor" if axis is None else "per-axis",
            "axis": "" if axis is None else axis,
            "scale": json.dumps(np.asarray(scale).tolist()),
            "zero_point": json.dumps(np.asarray(zero_point).tolist()),
            "sse": sse,
            "fp32_energy": energy,
            "mse": sse / reference.size,
            "nmse": sse / energy,
            "max_abs_error": float(np.max(np.abs(difference))),
        })

    per_group = []
    for group in GROUPS:
        totals = group_totals[group]
        if totals["weights"] == 0 or totals["energy"] == 0:
            raise ValueError(f"그룹 weight 또는 에너지가 비어 있습니다: {group}")
        per_group.append({
            "group": group,
            "weight_tensors": totals["weights"],
            "elements": totals["elements"],
            "sse": totals["sse"],
            "fp32_energy": totals["energy"],
            "mse": totals["sse"] / totals["elements"],
            "nmse": totals["sse"] / totals["energy"],
        })
    return per_weight, per_group, plot_values


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(output_dir, per_group, plot_values):
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for group, ax in zip(GROUPS, axes.flat):
        values = plot_values[group]
        original = np.concatenate([pair[0] for pair in values])
        restored = np.concatenate([pair[1] for pair in values])
        bins = np.linspace(min(original.min(), restored.min()),
                           max(original.max(), restored.max()), 81)
        ax.hist(original, bins=bins, density=True, histtype="step", label="FP32")
        ax.hist(restored, bins=bins, density=True, histtype="step", label="QDQ restored")
        ax.set_title(group)
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Density")
        ax.legend()
    figure.suptitle("Weight distributions: FP32 ONNX vs INT8 dequantized")
    figure.savefig(output_dir / "weight_distributions.png", dpi=160)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    values = [row["nmse"] for row in per_group]
    bars = ax.bar(GROUPS, values)
    ax.set_ylabel("NMSE = total squared error / total FP32 weight energy")
    ax.set_title("Weight quantization error by analysis group")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"{value:.2e}", ha="center", va="bottom", fontsize=8)
    figure.savefig(output_dir / "group_nmse.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="FP32 ONNX와 INT8 QDQ weight 오차 분석")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_hash = sha256_file(FP32_PATH)
    mapping = load_mapping(fp32_hash)
    fp32_model = onnx.load(FP32_PATH)
    int8_model = onnx.load(INT8_PATH)
    onnx.checker.check_model(fp32_model)
    onnx.checker.check_model(int8_model)

    per_weight, per_group, plot_values = calculate_errors(
        mapping, fp32_model, int8_model
    )
    save_csv(output_dir / "weight_errors.csv", per_weight)
    save_csv(output_dir / "group_errors.csv", per_group)
    save_plots(output_dir, per_group, plot_values)

    metadata = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "fp32_onnx_sha256": fp32_hash,
        "int8_onnx_sha256": sha256_file(INT8_PATH),
        "mapping_sha256": sha256_file(MAPPING_PATH),
        "weight_tensors": len(per_weight),
        "groups": list(GROUPS),
        "mse_definition": "sum((FP32 - dequantized)^2) / number of elements",
        "nmse_definition": "sum((FP32 - dequantized)^2) / sum(FP32^2)",
        "group_aggregation": "sum layer SSE / sum layer FP32 energy",
        "excluded": "Add has no weight; bias is INT32 and excluded",
        "software": {"numpy": np.__version__, "onnx": onnx.__version__},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("M6 weight quantization error (FP32 ONNX vs QDQ restored):")
    for row in per_group:
        print(f"{row['group']}: {row['weight_tensors']} weights, NMSE={row['nmse']:.6e}")
    print(f"결과 저장: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
