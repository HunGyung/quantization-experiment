"""Evaluate the three fixed M9 models once on CIFAR-10 test images."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

from data import SPLIT_PATH, make_data_loader
from evaluation import cpu_model_name, model_files, sha256_file
from evaluate_selective_int8 import (
    CANDIDATES,
    DEFAULT_OUTPUT_DIR as M8_RESULT_DIR,
    FROZEN_CONFIG_PATH,
    FROZEN_MODEL_PATH,
)
from export_onnx import load_onnx
from quantize_full_int8 import FP32_PATH, INT8_PATH, MAPPING_PATH
from quantize_selective_int8 import MANIFEST_PATH
from runtime_config import session_configuration


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "generated" / "m9"
MODEL_PATHS = {
    "fp32": FP32_PATH,
    "full_int8": INT8_PATH,
    "selective_int8": FROZEN_MODEL_PATH,
}


def check_frozen_models(m8_output_dir):
    """Check the M8 decision and model files before opening the test dataset."""
    comparison_path = m8_output_dir / "benchmark_comparison.json"
    required = [FROZEN_CONFIG_PATH, comparison_path, MANIFEST_PATH,
                MAPPING_PATH, SPLIT_PATH, *MODEL_PATHS.values()]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"M9 시작 전 필요한 파일이 없습니다: {path}")

    config = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    candidate = config["candidate"]
    if candidate not in CANDIDATES:
        raise ValueError(f"동결된 Selective 후보를 알 수 없습니다: {candidate}")
    if (config["keep_groups"] != list(CANDIDATES[candidate])
            or config["comparison_sha256"] != sha256_file(comparison_path)
            or config["candidate_manifest_sha256"] != sha256_file(MANIFEST_PATH)
            or config["mapping_sha256"] != sha256_file(MAPPING_PATH)
            or config["split_sha256"] != sha256_file(SPLIT_PATH)):
        raise ValueError("동결 설정과 M8 결과 또는 데이터 split이 다릅니다.")

    source = comparison["source"]
    if (comparison["mode"] != "benchmark"
            or comparison["test_split_used"] is not False
            or comparison["recommended_candidate"] != candidate
            or comparison["selection_rule"] != config["selection_rule"]
            or source["fp32_onnx_sha256"] != config["fp32_onnx_sha256"]
            or source["candidate_manifest_sha256"] != config["candidate_manifest_sha256"]
            or source["mapping_sha256"] != config["mapping_sha256"]
            or source["split_sha256"] != config["split_sha256"]
            or source["quantization_settings"] != config["quantization_settings"]
            or manifest["fp32_onnx_sha256"] != config["fp32_onnx_sha256"]
            or manifest["split_sha256"] != config["split_sha256"]
            or manifest["settings"] != config["quantization_settings"]):
        raise ValueError("M8 benchmark와 동결 설정의 출처가 다릅니다.")
    environment = comparison["benchmark_environment"]
    if (environment["cpu_model"] != cpu_model_name()
            or environment["onnxruntime_version"] != ort.__version__):
        raise ValueError("현재 CPU 또는 ONNX Runtime 버전이 M8 benchmark와 다릅니다.")

    candidate_entries = [entry for entry in manifest["candidates"]
                         if entry["candidate"] == candidate]
    benchmark_rows = {row["model_id"]: row for row in comparison["models"]}
    if len(candidate_entries) != 1 or candidate not in benchmark_rows:
        raise ValueError("선택된 후보가 M8 기록에 없습니다.")
    selected_row = benchmark_rows[candidate]
    if (candidate_entries[0]["onnx_sha256"] != config["candidate_onnx_sha256"]
            or candidate_entries[0]["keep_groups"] != config["keep_groups"]
            or selected_row["model_sha256"] != config["candidate_onnx_sha256"]
            or selected_row["validation_accuracy_percent"]
            != config["selected_validation_accuracy_percent"]
            or selected_row["latency_median_ms"]
            != config["selected_latency_median_ms"]):
        raise ValueError("동결 모델과 선택 근거가 다릅니다.")

    files = {name: model_files(path) for name, path in MODEL_PATHS.items()}
    expected_hashes = {
        "fp32": config["fp32_onnx_sha256"],
        "full_int8": source["full_int8_onnx_sha256"],
        "selective_int8": config["candidate_onnx_sha256"],
    }
    for name, expected in expected_hashes.items():
        if files[name]["onnx_sha256"] != expected:
            raise ValueError(f"{name} ONNX가 M8에서 확정한 파일과 다릅니다.")
        benchmark_name = candidate if name == "selective_int8" else name
        if (benchmark_rows[benchmark_name]["model_sha256"] != expected
                or benchmark_rows[benchmark_name]["total_size_bytes"]
                != files[name]["total_size_bytes"]):
            raise ValueError(f"{name} 모델 정보가 M8 benchmark와 다릅니다.")
    return config, comparison_path, comparison, files


def evaluate_test_accuracy(session, test_loader):
    input_name = session.get_inputs()[0].name
    correct = 0
    total = 0
    for images, labels in test_loader:
        logits = session.run(None, {input_name: images.numpy()})[0]
        if logits.shape != (len(labels), 10):
            raise ValueError(f"예상 밖의 출력 shape: {logits.shape}")
        correct += int(np.count_nonzero(np.argmax(logits, axis=1) == labels.numpy()))
        total += len(labels)
    if total != 10_000:
        raise ValueError(f"Test 이미지 수가 10,000장이 아닙니다: {total}")
    return {"sample_count": total, "correct_count": correct,
            "accuracy_percent": 100 * correct / total}


def main():
    parser = argparse.ArgumentParser(description="M9 최종 CIFAR-10 test 정확도")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--m8-output-dir", type=Path, default=M8_RESULT_DIR)
    args = parser.parse_args()
    json_path = args.output_dir / "test_comparison.json"
    csv_path = args.output_dir / "test_comparison.csv"
    if json_path.exists() or csv_path.exists():
        raise FileExistsError("M9 test 결과가 이미 있습니다. 기존 결과를 확인하세요.")

    config, comparison_path, comparison, files = check_frozen_models(args.m8_output_dir)
    print(f"동결된 Selective 후보: {config['candidate']} "
          f"(FP32 유지: {', '.join(config['keep_groups'])})")
    _, _, _, test_loader = make_data_loader()
    if len(test_loader.dataset) != 10_000:
        raise ValueError("Test 데이터는 정확히 10,000장이어야 합니다.")

    benchmark_rows = {row["model_id"]: row for row in comparison["models"]}
    rows = []
    runtime = None
    for name, path in MODEL_PATHS.items():
        session = load_onnx(path)
        settings = session_configuration(session)
        if settings != comparison["benchmark_environment"]["runtime"]:
            raise ValueError(f"ONNX Runtime 설정이 M8 benchmark와 다릅니다: {name}")
        if runtime is None:
            runtime = settings
        elif settings != runtime:
            raise ValueError(f"모델 간 ONNX Runtime 설정이 다릅니다: {name}")
        accuracy = evaluate_test_accuracy(session, test_loader)
        benchmark = benchmark_rows[config["candidate"] if name == "selective_int8"
                                   else name]
        row = {
            "model": name,
            "test_samples": accuracy["sample_count"],
            "test_correct": accuracy["correct_count"],
            "test_accuracy_percent": accuracy["accuracy_percent"],
            "accuracy_drop_pp_vs_fp32": None,
            "onnx_sha256": files[name]["onnx_sha256"],
            "total_size_bytes": files[name]["total_size_bytes"],
            "latency_median_ms_from_m8": benchmark["latency_median_ms"],
            "throughput_images_per_second_from_m8": benchmark["throughput_images_per_second"],
            "peak_ram_mib_from_m8": benchmark["peak_ram_mib"],
        }
        rows.append(row)
        print(f"{name}: test {accuracy['correct_count']}/10,000, "
              f"accuracy={accuracy['accuracy_percent']:.2f}%")

    baseline_correct = rows[0]["test_correct"]
    for row in rows:
        row["accuracy_drop_pp_vs_fp32"] = (
            100 * (baseline_correct - row["test_correct"]) / 10_000
        )
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CIFAR-10 official test split",
        "test_samples": 10_000,
        "split_sha256": config["split_sha256"],
        "frozen_config_sha256": sha256_file(FROZEN_CONFIG_PATH),
        "m8_benchmark_comparison_sha256": sha256_file(comparison_path),
        "selected_candidate_before_test": config["candidate"],
        "selective_keep_groups": config["keep_groups"],
        "runtime": runtime,
        "benchmark_note": "CPU performance values are reused from M8 validation-input benchmarks.",
        "models": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_json = args.output_dir / "test_comparison.tmp.json"
    temporary_csv = args.output_dir / "test_comparison.tmp.csv"
    try:
        with temporary_csv.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_csv.replace(csv_path)
        try:
            temporary_json.replace(json_path)
        except Exception:
            csv_path.unlink(missing_ok=True)
            raise
    finally:
        temporary_csv.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
    print(f"M9 결과: {json_path}")


if __name__ == "__main__":
    main()
