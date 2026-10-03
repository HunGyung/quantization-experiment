# CIFAR-10 ResNet-18 INT8 PTQ 실험

FP32 ResNet-18을 기준으로 ONNX Runtime CPU 환경에서 Static PTQ의 정확도와 효율을 비교하고, ResNet stage별 양자화 민감도를 분석한다.

## 로컬 준비

`data/`에는 CIFAR-10과 `split_indices.npz`, `checkpoints/`에는 `best_resnet18.pt`가 필요하다. 이 두 폴더는 Git에서 제외한다. ONNX 파일은 아래 명령으로 생성할 수 있다.

```powershell
.\.venv\Scripts\python.exe export_onnx.py
.\.venv\Scripts\python.exe graph_mapping.py
```

M3 검증 수치와 모델 해시는 [m3_export_verification.md](m3_export_verification.md)에 기록했다. 그래프 매핑은 [onnx_graph_mapping.csv](onnx_graph_mapping.csv)에 있다.

## M4 FP32 기준 측정

최종 비교에 사용할 데스크탑에서 실행한다.

```powershell
.\.venv\Scripts\python.exe evaluation.py checkpoints\onnx_resnet18.onnx --output results\fp32_desktop.json
```

평가 도구는 validation 정확도, ONNX 및 external data 파일 크기, batch 1 latency, batch 32 throughput, 별도 프로세스의 최대 RAM을 JSON으로 저장한다. 두 시간 지표는 각각 100회 warm-up 뒤 1,000회 측정하며 원시 시간도 저장한다. CPUExecutionProvider, intra-op 6 threads, inter-op 1 thread, `ORT_ENABLE_ALL` 최적화를 모든 모델에 적용한다. 시간 측정에는 준비된 입력의 `session.run` 호출만 포함한다. 메모리 값에는 Python·ONNX Runtime·모델 초기화와 추론을 포함한 작업 프로세스 전체가 들어간다.

이후 INT8 모델은 같은 데스크탑에서 측정하고 FP32 결과 파일을 참조한다.

```powershell
.\.venv\Scripts\python.exe evaluation.py checkpoints\full_int8.onnx --baseline-result results\fp32_desktop.json --output results\full_int8_desktop.json
```

인자 없이 `evaluation.py`를 실행하면 FP32 ONNX를 측정하고 `results/generated/`에 시간표시가 붙은 로컬 결과를 저장한다. 이 폴더는 Git에서 제외한다.
