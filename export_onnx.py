import torch
import onnx

from pathlib import Path

from model import load_model
from runtime_config import create_cpu_session


ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT / "checkpoints" / "onnx_resnet18.onnx"


def export_onnx():
    model = load_model()
    model.eval()

    ex_inputs = torch.randn(2, 3, 32, 32)

    onnx_program = torch.onnx.export(
        model,
        (ex_inputs,),
        dynamo=True,
        dynamic_shapes=({0: torch.export.Dim("batch", min=1)},),
        input_names=["images"],
        output_names=["logits"],
    )
    onnx_program.save(ONNX_PATH)


def load_onnx(model_path):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    session = create_cpu_session(model_path)

    return session


def main():
    if ONNX_PATH.exists() is False:
        export_onnx()

    session = load_onnx(ONNX_PATH)

    input_name = session.get_inputs()[0].name
    for batch_size in (1, 32):
        images = torch.randn(batch_size, 3, 32, 32).numpy()
        outputs = session.run(None, {input_name: images})[0]
        print(f"batch={batch_size}: output shape={outputs.shape}")
        # print("output:", outputs)


if __name__ == "__main__":
    main()
