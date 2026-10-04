import csv
import hashlib
from pathlib import Path

import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from data import SPLIT_PATH, make_data_loader
from export_onnx import load_onnx


ROOT = Path(__file__).resolve().parent
FP32_PATH = ROOT / "checkpoints" / "onnx_resnet18.onnx"
INT8_PATH = ROOT / "checkpoints" / "full_int8.onnx"
MAPPING_PATH = ROOT / "onnx_graph_mapping.csv"


class CIFAR10CalibrationReader(CalibrationDataReader):
    def __init__(self, cal_loader, input_name):
        self.cal_loader = cal_loader
        self.input_name = input_name
        self.rewind()

    def get_next(self):
        try:
            images, _ = next(self.iterator)
        except StopIteration:
            return None

        return {self.input_name: images.numpy()}

    def rewind(self):
        self.iterator = iter(self.cal_loader)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_fp32_model():
    if not FP32_PATH.is_file():
        raise FileNotFoundError(FP32_PATH)

    with MAPPING_PATH.open(newline="", encoding="utf-8-sig") as file:
        mapped_hashes = {row["onnx_sha256"] for row in csv.DictReader(file)}

    model_hash = sha256(FP32_PATH)
    if mapped_hashes != {model_hash}:
        raise ValueError("FP32 ONNX와 그래프 매핑표의 SHA-256이 다릅니다.")

    return model_hash


def quantize_full_int8():
    model_hash = check_fp32_model()
    _, _, cal_loader, _ = make_data_loader()
    if len(cal_loader.dataset) != 1000:
        raise ValueError("Calibration 데이터는 정확히 1,000장이어야 합니다.")

    fp32_session = load_onnx(FP32_PATH)
    input_name = fp32_session.get_inputs()[0].name
    reader = CIFAR10CalibrationReader(cal_loader, input_name)

    INT8_PATH.parent.mkdir(parents=True, exist_ok=True)
    # M3에서 확인한 노드 매핑을 유지하도록 추가 그래프 전처리는 하지 않는다.
    quantize_static(
        model_input=FP32_PATH,
        model_output=INT8_PATH,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "Gemm", "Add"],
        per_channel=False,
        reduce_range=False,
        calibration_providers=["CPUExecutionProvider"],
    )

    int8_model = onnx.load(INT8_PATH)
    onnx.checker.check_model(int8_model)
    int8_session = load_onnx(INT8_PATH)
    sample_images = next(iter(cal_loader))[0]
    for batch_size in (1, 32):
        sample = sample_images[:batch_size].numpy()
        output = int8_session.run(None, {input_name: sample})[0]
        print(f"batch {batch_size} 출력 shape: {output.shape}")

    qdq_count = sum(
        node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        for node in int8_model.graph.node
    )
    int8_initializer_count = sum(
        tensor.data_type == onnx.TensorProto.INT8
        for tensor in int8_model.graph.initializer
    )
    print(f"FP32 ONNX SHA-256: {model_hash}")
    print(f"Calibration: {len(cal_loader.dataset)}장, split SHA-256: {sha256(SPLIT_PATH)}")
    print("설정: QDQ, MinMax, activation/weight QInt8, Conv/Gemm/Add, per_channel=False")
    print(f"INT8 모델: {INT8_PATH}")
    print(
        f"Q/DQ 노드: {qdq_count}개, "
        f"INT8 initializer(가중치와 zero-point 포함): {int8_initializer_count}개"
    )


if __name__ == "__main__":
    quantize_full_int8()
