import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from data import SPLIT_PATH, make_data_loader
from export_onnx import load_onnx
from model import load_model
from runtime_config import session_configuration


ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT / "checkpoints" / "onnx_resnet18.onnx"


def measure_peak_memory(model_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "memory_worker.py"), str(Path(model_path).resolve())],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(model_path):
    """Include every external tensor data file used by the ONNX graph."""
    model_path = Path(model_path).resolve(strict=True)
    model = onnx.load(str(model_path), load_external_data=False)
    external_paths = set()

    def visit_tensor(tensor):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            return
        metadata = {item.key: item.value for item in tensor.external_data}
        location = metadata.get("location")
        if not location:
            raise ValueError(f"External tensor has no location: {tensor.name}")
        external_paths.add((model_path.parent / location).resolve(strict=True))

    def visit_sparse_tensor(tensor):
        visit_tensor(tensor.values)
        visit_tensor(tensor.indices)

    def visit_attribute(attribute):
        kind = attribute.type
        if kind == onnx.AttributeProto.TENSOR:
            visit_tensor(attribute.t)
        elif kind == onnx.AttributeProto.TENSORS:
            for tensor in attribute.tensors:
                visit_tensor(tensor)
        elif kind == onnx.AttributeProto.SPARSE_TENSOR:
            visit_sparse_tensor(attribute.sparse_tensor)
        elif kind == onnx.AttributeProto.SPARSE_TENSORS:
            for tensor in attribute.sparse_tensors:
                visit_sparse_tensor(tensor)
        elif kind == onnx.AttributeProto.GRAPH:
            visit_graph(attribute.g)
        elif kind == onnx.AttributeProto.GRAPHS:
            for graph in attribute.graphs:
                visit_graph(graph)

    def visit_nodes(nodes):
        for node in nodes:
            for attribute in node.attribute:
                visit_attribute(attribute)

    def visit_graph(graph):
        for tensor in graph.initializer:
            visit_tensor(tensor)
        for tensor in graph.sparse_initializer:
            visit_sparse_tensor(tensor)
        visit_nodes(graph.node)

    visit_graph(model.graph)
    for function in model.functions:
        visit_nodes(function.node)
        for attribute in function.attribute_proto:
            visit_attribute(attribute)
    for training in model.training_info:
        visit_graph(training.initialization)
        visit_graph(training.algorithm)

    files = [model_path, *sorted(external_paths)]
    return {
        "path": str(model_path),
        "onnx_sha256": sha256_file(model_path),
        "onnx_size_bytes": model_path.stat().st_size,
        "total_size_bytes": sum(path.stat().st_size for path in files),
        "external_data_files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(external_paths)
        ],
    }


def cpu_model_name():
    if sys.platform == "win32":
        try:
            import winreg

            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.uname().processor


def baseline_accuracy(baseline_result_path, validation, split_sha256):
    if baseline_result_path is None:
        return None, None
    baseline_path = Path(baseline_result_path).resolve(strict=True)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_validation = baseline["validation"]
    if baseline_validation["sample_count"] != validation["sample_count"]:
        raise ValueError("Baseline과 현재 validation sample 수가 다릅니다.")
    reference_split = baseline.get("dataset", {}).get("split_sha256")
    if reference_split and split_sha256 and reference_split != split_sha256:
        raise ValueError("Baseline과 현재 dataset split hash가 다릅니다.")
    return float(baseline_validation["accuracy_percent"]), str(baseline_path)


def evaluate_between_torch_and_onnx(val_loader):
    model = load_model()
    model.eval()

    session = load_onnx(ONNX_PATH)
    input_name = session.get_inputs()[0].name

    criterion = nn.CrossEntropyLoss()
    torch_loss_sum = 0.0
    onnx_loss_sum = 0.0
    torch_correct = 0
    onnx_correct = 0
    prediction_mismatches = 0
    max_logit_diff = 0.0
    logits_allclose = True
    total = 0

    # PyTorch FP32와 ONNX FP32의 출력 비교 허용 오차
    atol = 1e-3
    rtol = 1e-3

    with torch.inference_mode():
        for images, labels in val_loader:
            torch_logits = model(images)

            # ONNX Runtime 입력은 NumPy 배열이고, run()은 출력 배열의 리스트를 반환
            onnx_outputs = session.run(None, {input_name: images.numpy()})
            onnx_logits = torch.from_numpy(onnx_outputs[0])

            batch_size = labels.size(0)
            torch_loss_sum += criterion(torch_logits, labels).item() * batch_size
            onnx_loss_sum += criterion(onnx_logits, labels).item() * batch_size

            torch_predictions = torch_logits.argmax(dim=1)
            onnx_predictions = onnx_logits.argmax(dim=1)
            torch_correct += (torch_predictions == labels).sum().item()
            onnx_correct += (onnx_predictions == labels).sum().item()
            prediction_mismatches += (torch_predictions != onnx_predictions).sum().item()

            max_logit_diff = max(
                max_logit_diff,
                (torch_logits - onnx_logits).abs().max().item(),
            )
            logits_allclose = logits_allclose and torch.allclose(
                torch_logits, onnx_logits, atol=atol, rtol=rtol
            )
            total += batch_size

    print(
        f"PyTorch: val_loss={torch_loss_sum / total:.4f}, "
        f"val_accuracy={100 * torch_correct / total:.2f}%"
    )
    print(
        f"ONNX Runtime: val_loss={onnx_loss_sum / total:.4f}, "
        f"val_accuracy={100 * onnx_correct / total:.2f}%"
    )
    print(f"최대 logits 절댓값 차이: {max_logit_diff:.6g}")
    print(f"예측 불일치: {prediction_mismatches}/{total}")
    print(f"logits allclose (atol={atol}, rtol={rtol}): {logits_allclose}")


def evaluate_onnx(model_path, val_loader, result_path=None, baseline_result_path=None):
    model_path = Path(model_path).resolve(strict=True)
    model = model_files(model_path)
    print(f"파일 크기 (외부 데이터 포함): {model['total_size_bytes']} bytes")

    session = load_onnx(model_path)
    input_name = session.get_inputs()[0].name

    # Accuracy는 validation 전체에서 측정한다.
    criterion = nn.CrossEntropyLoss()
    onnx_loss_sum = 0.0
    onnx_correct = 0
    total = 0
    for images, labels in val_loader:
        onnx_logits = torch.from_numpy(
            session.run(None, {input_name: images.numpy()})[0]
        )
        batch_size = labels.size(0)
        onnx_loss_sum += criterion(onnx_logits, labels).item() * batch_size
        onnx_correct += (onnx_logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
    if total == 0:
        raise ValueError("Validation loader가 비어 있습니다.")
    validation = {
        "split": "validation",
        "sample_count": total,
        "loss": onnx_loss_sum / total,
        "accuracy_percent": 100 * onnx_correct / total,
        "accuracy_drop_percentage_points": None,
        "baseline_accuracy_percent": None,
        "baseline_result_path": None,
    }

    split_sha256 = sha256_file(SPLIT_PATH) if SPLIT_PATH.is_file() else None
    reference_accuracy, reference_path = baseline_accuracy(
        baseline_result_path, validation, split_sha256
    )
    if reference_accuracy is None and model_path == ONNX_PATH.resolve():
        # 명시된 FP32 기준 ONNX 파일 자체의 하락폭은 정의상 0%p이다.
        reference_accuracy = validation["accuracy_percent"]
    if reference_accuracy is not None:
        validation["baseline_accuracy_percent"] = reference_accuracy
        validation["baseline_result_path"] = reference_path
        validation["accuracy_drop_percentage_points"] = (
            reference_accuracy - validation["accuracy_percent"]
        )
    print(
        f"val_loss={validation['loss']:.4f}, "
        f"val_accuracy={validation['accuracy_percent']:.2f}%"
    )
    if validation["accuracy_drop_percentage_points"] is not None:
        print(
            "accuracy drop: "
            f"{validation['accuracy_drop_percentage_points']:.2f} %p"
        )

    # 같은 validation 첫 배치를 모든 모델에 재사용한다.
    first_images = next(iter(val_loader))[0]
    latency_input = first_images[:1].numpy()
    throughput_input = first_images[:32].numpy()
    if throughput_input.shape[0] != 32:
        raise ValueError("Throughput 측정에는 batch size 32가 필요합니다.")

    timing_scope = (
        "ONNX Runtime session.run call only; excludes data loading, "
        "preprocessing, and session initialization"
    )
    latency_feed = {input_name: latency_input}
    for _ in range(100):
        session.run(None, latency_feed)
    latency_ms = []
    for _ in range(1000):
        start_time = time.perf_counter()
        session.run(None, latency_feed)
        latency_ms.append((time.perf_counter() - start_time) * 1000)
    latency = {
        "batch_size": 1,
        "warmup_runs": 100,
        "measurement_runs": 1000,
        "mean_ms": float(np.mean(latency_ms)),
        "median_ms": float(np.median(latency_ms)),
        "raw_ms": latency_ms,
        "input_shape": list(latency_input.shape),
        "input_dtype": str(latency_input.dtype),
        "input_source": "first validation sample after evaluation transform",
        "timing_scope": timing_scope,
    }
    print(
        f"평균 latency: {latency['mean_ms']:.4f} ms | "
        f"latency 중앙값: {latency['median_ms']:.4f} ms"
    )

    throughput_feed = {input_name: throughput_input}
    for _ in range(100):
        session.run(None, throughput_feed)
    throughput_times_s = []
    for _ in range(1000):
        start_time = time.perf_counter()
        session.run(None, throughput_feed)
        throughput_times_s.append(time.perf_counter() - start_time)
    total_inference_time_s = sum(throughput_times_s)
    throughput = {
        "batch_size": 32,
        "warmup_runs": 100,
        "measurement_runs": 1000,
        "total_samples": 32 * 1000,
        "total_inference_time_s": total_inference_time_s,
        "images_per_second": 32 * 1000 / total_inference_time_s,
        "raw_times_s": throughput_times_s,
        "input_shape": list(throughput_input.shape),
        "input_dtype": str(throughput_input.dtype),
        "input_source": "first 32 validation samples after evaluation transform",
        "timing_scope": timing_scope,
    }
    print(f"throughput: {throughput['images_per_second']:.2f} images/s")

    # Peak working set은 모델별 새 프로세스에서 측정한다.
    memory = measure_peak_memory(model_path)
    print(
        f"peak process RAM (batch {memory['batch_size']}, "
        f"초기화 포함): {memory['peak_working_set_mib']:.2f} MiB"
    )

    result = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "dataset": {
            "split_path": str(SPLIT_PATH) if split_sha256 else None,
            "split_sha256": split_sha256,
        },
        "environment": {
            "cpu_model": cpu_model_name(),
            "logical_cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "onnxruntime_version": ort.__version__,
            "runtime": session_configuration(session),
        },
        "validation": validation,
        "latency": latency,
        "throughput": throughput,
        "memory": memory,
    }
    if result_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        result_path = (
            ROOT / "results" / "generated"
            / f"{model_path.stem}_{model['onnx_sha256'][:8]}_{timestamp}.json"
        )
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"결과 저장: {result_path.resolve()}")
    return result


def main():
    parser = argparse.ArgumentParser(description="ONNX validation 및 CPU 벤치마크")
    parser.add_argument("model_path", nargs="?", type=Path, default=ONNX_PATH)
    parser.add_argument(
        "--output", type=Path, help="결과 JSON 경로 (기본: results/generated/)"
    )
    parser.add_argument(
        "--baseline-result",
        type=Path,
        help="FP32 기준 결과 JSON 경로; 정확도 하락폭 계산에 사용",
    )
    args = parser.parse_args()
    _, val_loader, _, _ = make_data_loader()
    evaluate_onnx(
        args.model_path,
        val_loader,
        result_path=args.output,
        baseline_result_path=args.baseline_result,
    )


if __name__ == "__main__":
    main()
