# M3 FP32 ONNX export verification

Verified on 2026-10-03 with the local validation split (5,000 images) on an AMD Ryzen 5 5600U laptop.
This verifies export equivalence; it is not the final desktop CPU benchmark.

| Item | Result |
| --- | --- |
| FP32 checkpoint | `checkpoints/best_resnet18.pt` (epoch 177, seed 42) |
| Checkpoint SHA-256 | `495910c4b30f92dbe01df8a8a2085bb1acf28a9623fa7fd98e8da927b6cc53df` |
| FP32 ONNX | `checkpoints/onnx_resnet18.onnx` |
| ONNX SHA-256 | `57899ba9edd1a8297fd2e963836db34380166f38d0a1c48978adbe02692e3c45` |
| Split index SHA-256 | `91fc05b99cdf0fefd0deb179762c503035a6bdb386a0cb2b233f607862e00acb` |
| PyTorch validation | loss 0.4833, top-1 accuracy 94.00% |
| ONNX Runtime validation | loss 0.4833, top-1 accuracy 94.00% |
| Maximum absolute logit difference | 2.67029e-05 |
| Prediction mismatches | 0 / 5,000 |
| `torch.allclose` | True (`atol=1e-3`, `rtol=1e-3`) |
| ONNX batch 1 / 32 output shapes | `(1, 10)` / `(32, 10)` |
| ONNX graph mapping | `onnx_graph_mapping.csv`, 20 Conv + 8 Add + 1 Gemm nodes |

Versions: PyTorch 2.13.0+cu130, ONNX 1.23.1, ONNX Runtime 1.30.0.
The ONNX session used CPUExecutionProvider, 6 intra-op threads, 1 inter-op thread, and `ORT_ENABLE_ALL` optimization.

Reproduction commands (from the repository root with the local checkpoint, ONNX model, and CIFAR-10 data present):

```powershell
.\.venv\Scripts\python.exe export_onnx.py
.\.venv\Scripts\python.exe -c "from evaluation import evaluate_between_torch_and_onnx; from data import make_data_loader; evaluate_between_torch_and_onnx(make_data_loader()[1])"
.\.venv\Scripts\python.exe graph_mapping.py
```

The checkpoint and ONNX binary are intentionally excluded by `.gitignore`; the mapping CSV records the ONNX hash it belongs to.
