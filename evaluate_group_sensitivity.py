"""Measure validation accuracy for the six M7 single-group INT8 models."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import onnxruntime as ort

from data import SPLIT_PATH, make_data_loader
from evaluation import evaluate_validation, sha256_file
from export_onnx import load_onnx
from quantize_full_int8 import FP32_PATH, MAPPING_PATH
from quantize_group_int8 import (
    GROUP_SLUGS,
    MANIFEST_PATH,
    MODEL_PATHS,
    load_mapping,
    verify_group_model,
)


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "generated" / "m7"


def load_verified_models(rows, fp32_model, fp32_hash):
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"M7 manifest가 없습니다. 먼저 quantize_group_int8.py를 실행하세요: {MANIFEST_PATH}"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["fp32_onnx_sha256"] != fp32_hash:
        raise ValueError("M7 모델을 만든 FP32 ONNX가 현재 기준 모델과 다릅니다.")
    if manifest["mapping_sha256"] != sha256_file(MAPPING_PATH):
        raise ValueError("M7 모델을 만든 그래프 매핑표가 현재 파일과 다릅니다.")
    if manifest["split_sha256"] != sha256_file(SPLIT_PATH):
        raise ValueError("M7 모델을 만든 calibration/validation split이 현재와 다릅니다.")
    expected_settings = {
        "quant_format": "QDQ",
        "calibrate_method": "MinMax",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "op_types_to_quantize": ["Conv", "Gemm", "Add"],
        "per_channel": False,
        "reduce_range": False,
        "calibration_providers": ["CPUExecutionProvider"],
    }
    if manifest["calibration_samples"] != 1000 or manifest["settings"] != expected_settings:
        raise ValueError("M7 manifest의 calibration 수 또는 양자화 설정이 M5와 다릅니다.")

    entries = manifest["models"]
    if len(entries) != len(GROUP_SLUGS) or {
        entry["group"] for entry in entries
    } != set(GROUP_SLUGS):
        raise ValueError("M7 manifest에 여섯 그룹 모델이 정확히 한 개씩 있어야 합니다.")

    verified = {}
    for entry in entries:
        group = entry["group"]
        model_path = MODEL_PATHS[group]
        expected_nodes = [row["node_name"] for row in rows if row["group"] == group]
        if entry["file"] != model_path.name or entry["target_nodes"] != expected_nodes:
            raise ValueError(f"M7 manifest의 모델 또는 대상 노드가 다릅니다: {group}")
        if not model_path.is_file() or sha256_file(model_path) != entry["onnx_sha256"]:
            raise ValueError(f"M7 모델 파일이 manifest와 다릅니다: {group}")
        graph_report = verify_group_model(model_path, group, rows, fp32_model)
        verified[group] = {
            "path": model_path,
            "sha256": entry["onnx_sha256"],
            "target_nodes": expected_nodes,
            "graph_verification": graph_report,
        }
    return manifest, verified


def save_results(output_dir, manifest, verified, baseline, ranked):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in ranked:
        rows.append({
            "rank": item["rank"],
            "group": item["group"],
            "quantized_nodes": len(verified[item["group"]]["target_nodes"]),
            "fp32_accuracy_percent": baseline["accuracy_percent"],
            "fp32_correct": baseline["correct_count"],
            "group_accuracy_percent": item["accuracy_percent"],
            "group_correct": item["correct_count"],
            "correct_count_drop": item["correct_count_drop"],
            "sensitivity_percentage_points": item["sensitivity_pp"],
            "group_loss": item["loss"],
            "model_sha256": verified[item["group"]]["sha256"],
        })
    with (output_dir / "sensitivity.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "definition": "FP32 validation accuracy - single-group INT8 validation accuracy (percentage points)",
        "interpretation": (
            "Each model quantizes only the mapped group targets. Activation Q/DQ "
            "at group boundaries also affects this task-level measurement; "
            "weight NMSE alone does not determine sensitivity."
        ),
        "validation_split_sha256": manifest["split_sha256"],
        "fp32_onnx_sha256": manifest["fp32_onnx_sha256"],
        "mapping_sha256": manifest["mapping_sha256"],
        "quantization": {
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "calibration_samples": manifest["calibration_samples"],
            "settings": manifest["settings"],
            "software": manifest["software"],
        },
        "onnxruntime_version": ort.__version__,
        "baseline": baseline,
        "ranking": ranked,
        "most_sensitive_groups": [
            item["group"] for item in ranked
            if item["correct_count_drop"] == ranked[0]["correct_count_drop"]
        ],
    }
    (output_dir / "sensitivity.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    graph_records = {
        "fp32_onnx_sha256": manifest["fp32_onnx_sha256"],
        "mapping_sha256": manifest["mapping_sha256"],
        "models": [
            {
                "group": group,
                "model_path": str(verified[group]["path"]),
                "model_sha256": verified[group]["sha256"],
                "selected_nodes": verified[group]["target_nodes"],
                "graph_verification": verified[group]["graph_verification"],
            }
            for group in GROUP_SLUGS
        ],
    }
    (output_dir / "graph_verification.json").write_text(
        json.dumps(graph_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    figure, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    labels = [item["group"] for item in ranked]
    values = [item["sensitivity_pp"] for item in ranked]
    bars = ax.barh(labels, values)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("FP32 accuracy - single-group INT8 accuracy (% points)")
    ax.set_title("Single-group PTQ sensitivity: CIFAR-10 validation (n=5,000)")
    offset = max(max(abs(value) for value in values), 0.02) * 0.025
    for bar, value in zip(bars, values):
        x = value + offset if value >= 0 else value - offset
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}",
                ha="left" if value >= 0 else "right", va="center")
    ax.margins(x=0.15)
    figure.savefig(output_dir / "sensitivity.png", dpi=160)
    plt.close(figure)


def correct_count(measured):
    count = round(measured["accuracy_percent"] * measured["sample_count"] / 100)
    if abs(100 * count / measured["sample_count"] - measured["accuracy_percent"]) > 1e-9:
        raise ValueError("정확도가 validation 정답 개수와 일치하지 않습니다.")
    return count


def main():
    rows, fp32_model, fp32_hash = load_mapping()
    _, val_loader, cal_loader, _ = make_data_loader()
    if len(val_loader.dataset) != 5000 or len(cal_loader.dataset) != 1000:
        raise ValueError("Validation 5,000장과 calibration 1,000장을 확인하세요.")
    manifest, verified = load_verified_models(rows, fp32_model, fp32_hash)

    baseline = evaluate_validation(load_onnx(FP32_PATH), val_loader)
    if baseline["sample_count"] != 5000:
        raise ValueError("FP32 validation 평가에 사용된 이미지 수가 다릅니다.")
    baseline["correct_count"] = correct_count(baseline)

    results = []
    for group in GROUP_SLUGS:
        model_path = verified[group]["path"]
        measured = evaluate_validation(load_onnx(model_path), val_loader)
        if measured["sample_count"] != baseline["sample_count"]:
            raise ValueError(f"Validation 이미지 수가 다릅니다: {group}")
        group_correct = correct_count(measured)
        correct_drop = baseline["correct_count"] - group_correct
        sensitivity = 100 * correct_drop / baseline["sample_count"]
        results.append({
            "group": group,
            "model_path": str(model_path),
            "model_sha256": verified[group]["sha256"],
            "target_nodes": verified[group]["target_nodes"],
            "sample_count": measured["sample_count"],
            "accuracy_percent": measured["accuracy_percent"],
            "correct_count": group_correct,
            "correct_count_drop": correct_drop,
            "loss": measured["loss"],
            "sensitivity_pp": sensitivity,
        })
        print(f"{group}: accuracy={measured['accuracy_percent']:.2f}%, "
              f"sensitivity={sensitivity:+.2f} %p")

    ranked = sorted(results, key=lambda item: -item["correct_count_drop"])
    previous_drop = None
    for position, item in enumerate(ranked, start=1):
        if item["correct_count_drop"] != previous_drop:
            shared_rank = position
        item["rank"] = shared_rank
        previous_drop = item["correct_count_drop"]
    save_results(RESULT_DIR, manifest, verified, baseline, ranked)
    print(f"FP32 validation accuracy: {baseline['accuracy_percent']:.2f}%")
    print("민감도 순위 (accuracy drop 큰 순서):")
    for item in ranked:
        print(f"  {item['rank']}. {item['group']}: {item['sensitivity_pp']:+.2f} %p "
              f"({item['correct_count_drop']:+d}/5,000장)")
    print(f"결과 저장: {RESULT_DIR}")


if __name__ == "__main__":
    main()
