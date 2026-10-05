"""Compare M8 selective INT8 candidates using validation and optional CPU benchmarks."""

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import SPLIT_PATH, make_data_loader
from evaluate_group_sensitivity import correct_count
from evaluation import evaluate_onnx, evaluate_validation, model_files, sha256_file
from export_onnx import load_onnx
from quantize_full_int8 import FP32_PATH, INT8_PATH, MAPPING_PATH
from quantize_group_int8 import MANIFEST_PATH as M7_MANIFEST_PATH, load_mapping
from quantize_selective_int8 import (
    CANDIDATES,
    MANIFEST_PATH,
    MODEL_PATHS,
    PTQ_SETTINGS,
    verify_selective_model,
)
from runtime_config import (
    EXECUTION_MODE,
    GRAPH_OPTIMIZATION_LEVEL,
    INTER_OP_NUM_THREADS,
    INTRA_OP_NUM_THREADS,
    PROVIDER,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "results" / "generated" / "m8"
FROZEN_MODEL_PATH = ROOT / "checkpoints" / "selective_int8.onnx"
FROZEN_CONFIG_PATH = ROOT / "checkpoints" / "selective_int8_config.json"
MODEL_ORDER = ("fp32", "full_int8", *CANDIDATES)
SELECTION_RULE = (
    "Among the three selective candidates, choose highest validation correct count; "
    "then lowest batch-1 median latency; then highest batch-32 throughput; "
    "then smallest total model size; then lowest peak RAM; "
    "then the fixed candidate order in CANDIDATES."
)


def load_verified_models(rows, fp32_model, fp32_hash):
    """Reject stale models, changed calibration settings, or wrong graph scope."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"M8 manifest가 없습니다. 먼저 quantize_selective_int8.py를 실행하세요: {MANIFEST_PATH}"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["fp32_onnx_sha256"] != fp32_hash:
        raise ValueError("M8 모델을 만든 FP32 ONNX가 현재 기준 모델과 다릅니다.")
    if (not M7_MANIFEST_PATH.is_file()
            or manifest["m7_manifest_sha256"] != sha256_file(M7_MANIFEST_PATH)):
        raise ValueError("M8 모델을 만들 때 참조한 M7 manifest가 현재와 다릅니다.")
    if manifest["mapping_sha256"] != sha256_file(MAPPING_PATH):
        raise ValueError("M8 모델을 만든 M3 매핑표가 현재 파일과 다릅니다.")
    if manifest["split_sha256"] != sha256_file(SPLIT_PATH):
        raise ValueError("M8 모델을 만든 calibration/validation split이 현재와 다릅니다.")
    if manifest["calibration_samples"] != 1000 or manifest["settings"] != PTQ_SETTINGS:
        raise ValueError("M8 calibration 수 또는 양자화 설정이 M5와 다릅니다.")

    entries = manifest["candidates"]
    if len(entries) != len(CANDIDATES) or {
        entry["candidate"] for entry in entries
    } != set(CANDIDATES):
        raise ValueError("M8 manifest에 세 후보가 정확히 한 개씩 있어야 합니다.")

    verified = {}
    for entry in entries:
        candidate = entry["candidate"]
        keep_groups = CANDIDATES[candidate]
        path = MODEL_PATHS[candidate]
        target_nodes = [row["node_name"] for row in rows
                        if row["group"] not in keep_groups]
        kept_nodes = [row["node_name"] for row in rows
                      if row["group"] in keep_groups]
        if (entry["file"] != path.name
                or entry["keep_groups"] != list(keep_groups)
                or entry["target_nodes"] != target_nodes
                or entry["kept_nodes"] != kept_nodes):
            raise ValueError(f"M8 manifest의 후보 설정이 현재 M3 매핑과 다릅니다: {candidate}")
        if not path.is_file() or sha256_file(path) != entry["onnx_sha256"]:
            raise ValueError(f"M8 후보 모델 해시가 manifest와 다릅니다: {candidate}")
        graph_report = verify_selective_model(path, keep_groups, rows, fp32_model)
        verified[candidate] = {
            "path": path,
            "sha256": entry["onnx_sha256"],
            "keep_groups": list(keep_groups),
            "target_nodes": target_nodes,
            "kept_nodes": kept_nodes,
            "graph_verification": graph_report,
        }

    if not INT8_PATH.is_file():
        raise FileNotFoundError(INT8_PATH)
    full_report = verify_selective_model(INT8_PATH, (), rows, fp32_model)
    paths = {"fp32": FP32_PATH, "full_int8": INT8_PATH}
    paths.update({candidate: verified[candidate]["path"] for candidate in CANDIDATES})
    return manifest, verified, paths, full_report


def run_validation_only(paths, val_loader):
    results = {}
    for model_id in MODEL_ORDER:
        path = paths[model_id]
        measured = evaluate_validation(load_onnx(path), val_loader)
        results[model_id] = {
            "model": model_files(path),
            "validation": measured,
        }
        print(f"{model_id}: validation accuracy={measured['accuracy_percent']:.2f}%")
    return results


def check_benchmark_comparability(results, split_hash):
    reference = results["fp32"]
    for model_id in MODEL_ORDER:
        current = results[model_id]
        if current["dataset"]["split_sha256"] != split_hash:
            raise ValueError(f"Benchmark split hash가 다릅니다: {model_id}")
        if current["validation"]["sample_count"] != 5000:
            raise ValueError(f"Benchmark validation 이미지 수가 다릅니다: {model_id}")
        runtime = current["environment"]["runtime"]
        if (runtime["providers"] != [PROVIDER]
                or runtime["intra_op_num_threads"] != INTRA_OP_NUM_THREADS
                or runtime["inter_op_num_threads"] != INTER_OP_NUM_THREADS
                or runtime["graph_optimization_level"] != str(GRAPH_OPTIMIZATION_LEVEL)
                or runtime["execution_mode"] != str(EXECUTION_MODE)):
            raise ValueError(f"{model_id}: CPU benchmark 실행 설정이 M4와 다릅니다.")
        for key in ("cpu_model", "logical_cpu_count", "platform", "python_version",
                    "onnxruntime_version", "runtime"):
            if current["environment"][key] != reference["environment"][key]:
                raise ValueError(f"Benchmark 환경 {key}가 다릅니다: {model_id}")
        for metric, batch_size in (("latency", 1), ("throughput", 32)):
            current_metric = current[metric]
            reference_metric = reference[metric]
            if current_metric["batch_size"] != batch_size:
                raise ValueError(f"{model_id}: {metric} batch size가 다릅니다.")
            if (current_metric["warmup_runs"] != 100
                    or current_metric["measurement_runs"] != 1000):
                raise ValueError(f"{model_id}: {metric} 반복 횟수가 M4와 다릅니다.")
            expected_source = (
                "first validation sample after evaluation transform" if metric == "latency"
                else "first 32 validation samples after evaluation transform"
            )
            if (current_metric["input_shape"] != [batch_size, 3, 32, 32]
                    or current_metric["input_dtype"] != "float32"
                    or current_metric["input_source"] != expected_source
                    or current_metric["timing_scope"] != (
                        "ONNX Runtime session.run call only; excludes data loading, "
                        "preprocessing, and session initialization"
                    )):
                raise ValueError(f"{model_id}: {metric} 입력/측정 범위가 M4와 다릅니다.")
            for key in ("warmup_runs", "measurement_runs", "input_shape",
                        "input_dtype", "input_source", "timing_scope"):
                if current_metric[key] != reference_metric[key]:
                    raise ValueError(f"{model_id}: {metric} 측정 조건 {key}가 다릅니다.")
        for key in ("batch_size", "warmup_runs", "measurement_runs", "input", "runtime"):
            if current["memory"][key] != reference["memory"][key]:
                raise ValueError(f"{model_id}: 메모리 측정 조건 {key}가 다릅니다.")
        memory = current["memory"]
        if (memory["batch_size"] != 1 or memory["warmup_runs"] != 100
                or memory["measurement_runs"] != 1000
                or memory["input"] != {
                    "source": "fixed zero array",
                    "shape": [1, 3, 32, 32],
                    "dtype": "float32",
                } or memory["runtime"] != runtime
                or memory["scope"] != "worker process startup, session initialization, and inference"):
            raise ValueError(f"{model_id}: 메모리 측정 조건이 M4와 다릅니다.")


def run_benchmarks(paths, val_loader, output_dir, split_hash):
    benchmark_dir = output_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = benchmark_dir / "fp32.json"
    results = {}
    for model_id in MODEL_ORDER:
        result_path = benchmark_dir / f"{model_id}.json"
        print(f"M8 benchmark: {model_id}")
        results[model_id] = evaluate_onnx(
            paths[model_id], val_loader, result_path=result_path,
            baseline_result_path=None if model_id == "fp32" else baseline_path,
        )
    check_benchmark_comparability(results, split_hash)
    return results


def make_comparison_rows(results, benchmark):
    baseline = results["fp32"]
    baseline_validation = baseline["validation"]
    if baseline_validation["sample_count"] != 5000:
        raise ValueError("FP32 validation 이미지 수가 5,000장이 아닙니다.")
    baseline_correct = correct_count(baseline_validation)
    baseline_size = baseline["model"]["total_size_bytes"]
    rows = []
    for model_id in MODEL_ORDER:
        result = results[model_id]
        validation = result["validation"]
        if validation["sample_count"] != 5000:
            raise ValueError(f"Validation 이미지 수가 다릅니다: {model_id}")
        correct = correct_count(validation)
        size = result["model"]["total_size_bytes"]
        row = {
            "model_id": model_id,
            "keep_groups": ";".join(CANDIDATES.get(model_id, ())),
            "model_sha256": result["model"]["onnx_sha256"],
            "validation_samples": 5000,
            "validation_correct": correct,
            "validation_accuracy_percent": validation["accuracy_percent"],
            "accuracy_drop_pp": 100 * (baseline_correct - correct) / 5000,
            "validation_loss": validation["loss"],
            "total_size_bytes": size,
            "compression_ratio_vs_fp32": baseline_size / size,
            "latency_median_ms": None,
            "latency_mean_ms": None,
            "latency_speedup_vs_fp32": None,
            "throughput_images_per_second": None,
            "throughput_ratio_vs_fp32": None,
            "peak_ram_mib": None,
            "peak_ram_reduction_percent_vs_fp32": None,
        }
        if benchmark:
            reference = results["fp32"]
            row.update({
                "latency_median_ms": result["latency"]["median_ms"],
                "latency_mean_ms": result["latency"]["mean_ms"],
                "latency_speedup_vs_fp32": (
                    reference["latency"]["median_ms"] / result["latency"]["median_ms"]
                ),
                "throughput_images_per_second": result["throughput"]["images_per_second"],
                "throughput_ratio_vs_fp32": (
                    result["throughput"]["images_per_second"] /
                    reference["throughput"]["images_per_second"]
                ),
                "peak_ram_mib": result["memory"]["peak_working_set_mib"],
                "peak_ram_reduction_percent_vs_fp32": 100 * (
                    reference["memory"]["peak_working_set_mib"] -
                    result["memory"]["peak_working_set_mib"]
                ) / reference["memory"]["peak_working_set_mib"],
            })
        rows.append(row)
    return rows


def recommend_candidate(rows):
    """Apply the recorded accuracy-first rule to benchmarked selective candidates."""
    by_id = {row["model_id"]: row for row in rows}
    def key(candidate):
        row = by_id[candidate]
        return (
            -row["validation_correct"],
            row["latency_median_ms"],
            -row["throughput_images_per_second"],
            row["total_size_bytes"],
            row["peak_ram_mib"],
            list(CANDIDATES).index(candidate),
        )
    return min(CANDIDATES, key=key)


def save_comparison(output_dir, rows, results, manifest, verified, full_report,
                    benchmark, recommendation):
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "benchmark" if benchmark else "validation"
    csv_path = output_dir / f"{prefix}_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    graph_record = {
        "fp32_onnx_sha256": manifest["fp32_onnx_sha256"],
        "mapping_sha256": manifest["mapping_sha256"],
        "full_int8": full_report,
        "candidates": {
            name: verified[name]["graph_verification"] for name in CANDIDATES
        },
    }
    (output_dir / "graph_verification.json").write_text(
        json.dumps(graph_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": prefix,
        "validation_only_is_not_final_selection": not benchmark,
        "test_split_used": False,
        "source": {
            "fp32_onnx_sha256": manifest["fp32_onnx_sha256"],
            "m7_manifest_sha256": manifest["m7_manifest_sha256"],
            "full_int8_onnx_sha256": results["full_int8"]["model"]["onnx_sha256"],
            "mapping_sha256": manifest["mapping_sha256"],
            "split_sha256": manifest["split_sha256"],
            "candidate_manifest_sha256": sha256_file(MANIFEST_PATH),
            "calibration_samples": manifest["calibration_samples"],
            "quantization_settings": manifest["settings"],
            "quantization_software": manifest["software"],
        },
        "selection_rule": SELECTION_RULE,
        "recommended_candidate": recommendation,
        "models": rows,
        "benchmark_environment": results["fp32"].get("environment") if benchmark else None,
        "benchmark_raw_result_files": (
            {name: str(output_dir / "benchmarks" / f"{name}.json")
             for name in MODEL_ORDER} if benchmark else None
        ),
        "benchmark_raw_result_sha256": (
            {name: sha256_file(output_dir / "benchmarks" / f"{name}.json")
             for name in MODEL_ORDER} if benchmark else None
        ),
    }
    json_path = output_dir / f"{prefix}_comparison.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    figure, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    if benchmark:
        for row in rows:
            ax.scatter(row["latency_median_ms"], row["validation_accuracy_percent"])
            ax.annotate(row["model_id"],
                        (row["latency_median_ms"], row["validation_accuracy_percent"]),
                        xytext=(5, 4), textcoords="offset points")
        ax.set_xlabel("Batch-1 median latency (ms)")
        ax.set_ylabel("Validation top-1 accuracy (%)")
        ax.set_title("M8 accuracy-latency comparison (same CPU protocol)")
        figure.savefig(output_dir / "accuracy_latency.png", dpi=160)
    else:
        bars = ax.barh([row["model_id"] for row in rows],
                       [row["accuracy_drop_pp"] for row in rows])
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Accuracy drop vs FP32 (% points)")
        ax.set_title("M8 validation accuracy preview (n=5,000)")
        for bar, row in zip(bars, rows):
            value = row["accuracy_drop_pp"]
            ax.text(value, bar.get_y() + bar.get_height() / 2,
                    f" {value:+.2f}", va="center")
        ax.margins(x=0.15)
        figure.savefig(output_dir / "validation_accuracy.png", dpi=160)
    plt.close(figure)
    return json_path


def load_benchmark_for_freeze(output_dir, manifest, paths, expected_hashes):
    """Recheck saved benchmark data before copying the chosen model."""
    comparison_path = output_dir / "benchmark_comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError("먼저 --benchmark로 다섯 모델을 측정하세요.")
    summary = json.loads(comparison_path.read_text(encoding="utf-8"))
    source = summary["source"]
    if (summary["mode"] != "benchmark" or summary["test_split_used"] is not False
            or summary["selection_rule"] != SELECTION_RULE
            or source["fp32_onnx_sha256"] != manifest["fp32_onnx_sha256"]
            or source["m7_manifest_sha256"] != manifest["m7_manifest_sha256"]
            or source["full_int8_onnx_sha256"] != expected_hashes["full_int8"]
            or source["mapping_sha256"] != manifest["mapping_sha256"]
            or source["split_sha256"] != manifest["split_sha256"]
            or source["candidate_manifest_sha256"] != sha256_file(MANIFEST_PATH)
            or source["calibration_samples"] != manifest["calibration_samples"]
            or source["quantization_settings"] != manifest["settings"]):
        raise ValueError("저장된 benchmark 비교 기록의 입력·설정이 현재 모델과 다릅니다.")

    results = {}
    for model_id in MODEL_ORDER:
        result_path = output_dir / "benchmarks" / f"{model_id}.json"
        if (not result_path.is_file()
                or sha256_file(result_path) != summary["benchmark_raw_result_sha256"][model_id]):
            raise ValueError(f"Benchmark 원시 결과 파일이 바뀌었습니다: {model_id}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        current_model = model_files(paths[model_id])
        if (result["model"]["onnx_sha256"] != expected_hashes[model_id]
                or result["model"]["onnx_sha256"] != current_model["onnx_sha256"]
                or result["model"]["total_size_bytes"] != current_model["total_size_bytes"]
                or result["model"]["external_data_files"] != current_model["external_data_files"]):
            raise ValueError(f"Benchmark 모델 해시가 현재 파일과 다릅니다: {model_id}")
        results[model_id] = result
    check_benchmark_comparability(results, manifest["split_sha256"])
    comparison_rows = make_comparison_rows(results, benchmark=True)
    recommendation = recommend_candidate(comparison_rows)
    if (summary["models"] != comparison_rows
            or summary["recommended_candidate"] != recommendation):
        raise ValueError("저장된 비교표 또는 추천 후보가 원시 측정값과 다릅니다.")
    return comparison_path, comparison_rows, recommendation


def freeze_recommendation(candidate, manifest, comparison_path, rows):
    if FROZEN_MODEL_PATH.exists() or FROZEN_CONFIG_PATH.exists():
        raise FileExistsError("최종 Selective INT8이 이미 동결되어 있습니다.")
    source = MODEL_PATHS[candidate]
    if model_files(source)["external_data_files"]:
        raise ValueError("외부 tensor 파일이 있는 모델은 이 동결 명령에서 지원하지 않습니다.")
    source_hash = sha256_file(source)
    row = next(row for row in rows if row["model_id"] == candidate)
    if row["model_sha256"] != source_hash:
        raise ValueError("선택한 후보 모델이 비교표와 다릅니다.")
    temporary_model = FROZEN_MODEL_PATH.with_name("selective_int8.tmp.onnx")
    temporary_config = FROZEN_CONFIG_PATH.with_name("selective_int8_config.tmp.json")
    config = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate,
        "keep_groups": list(CANDIDATES[candidate]),
        "selection_rule": SELECTION_RULE,
        "candidate_onnx_sha256": source_hash,
        "comparison_sha256": sha256_file(comparison_path),
        "candidate_manifest_sha256": sha256_file(MANIFEST_PATH),
        "fp32_onnx_sha256": manifest["fp32_onnx_sha256"],
        "mapping_sha256": manifest["mapping_sha256"],
        "split_sha256": manifest["split_sha256"],
        "quantization_settings": manifest["settings"],
        "selected_validation_accuracy_percent": row["validation_accuracy_percent"],
        "selected_latency_median_ms": row["latency_median_ms"],
    }
    try:
        shutil.copyfile(source, temporary_model)
        if sha256_file(temporary_model) != source_hash:
            raise ValueError("최종 모델 복사 후 해시가 다릅니다.")
        temporary_config.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_model.replace(FROZEN_MODEL_PATH)
        try:
            temporary_config.replace(FROZEN_CONFIG_PATH)
        except Exception:
            FROZEN_MODEL_PATH.unlink(missing_ok=True)
            raise
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_config.unlink(missing_ok=True)
    print(f"최종 Selective INT8 동결: {candidate} → {FROZEN_MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="M8 Selective INT8 후보 비교")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--benchmark", action="store_true",
                      help="M4와 같은 CPU 지표를 FP32/Full INT8/후보 3개에 모두 측정")
    mode.add_argument("--freeze", action="store_true",
                      help="저장된 benchmark 비교를 검증한 뒤 추천 모델 동결")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.freeze and (FROZEN_MODEL_PATH.exists() or FROZEN_CONFIG_PATH.exists()):
        raise FileExistsError("최종 Selective INT8이 이미 동결되어 있습니다.")

    rows, fp32_model, fp32_hash = load_mapping()
    manifest, verified, paths, full_report = load_verified_models(
        rows, fp32_model, fp32_hash
    )
    expected_hashes = {"fp32": fp32_hash,
                       "full_int8": sha256_file(INT8_PATH),
                       **{name: verified[name]["sha256"] for name in CANDIDATES}}
    split_hash = sha256_file(SPLIT_PATH)
    if args.freeze:
        comparison_path, comparison_rows, recommendation = load_benchmark_for_freeze(
            args.output_dir, manifest, paths, expected_hashes
        )
        freeze_recommendation(recommendation, manifest, comparison_path,
                              comparison_rows)
        return

    _, val_loader, cal_loader, _ = make_data_loader()
    if len(val_loader.dataset) != 5000 or len(cal_loader.dataset) != 1000:
        raise ValueError("Validation 5,000장과 calibration 1,000장을 확인하세요.")
    if args.benchmark:
        results = run_benchmarks(paths, val_loader, args.output_dir, split_hash)
    else:
        results = run_validation_only(paths, val_loader)
    for model_id, result in results.items():
        if result["model"]["onnx_sha256"] != expected_hashes[model_id]:
            raise ValueError(f"평가한 모델 해시가 바뀌었습니다: {model_id}")
    comparison_rows = make_comparison_rows(results, args.benchmark)
    recommendation = recommend_candidate(comparison_rows) if args.benchmark else None
    comparison_path = save_comparison(
        args.output_dir, comparison_rows, results, manifest, verified,
        full_report, args.benchmark, recommendation,
    )
    if args.benchmark:
        print(f"사전 정의된 규칙의 추천 후보: {recommendation}")
    else:
        print("효율 비교와 최종 선택 전 validation 확인 결과입니다.")
    print(f"M8 비교표: {comparison_path}")


if __name__ == "__main__":
    main()
