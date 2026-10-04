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

최종 비교에 사용할 장비를 하나 정해 FP32와 모든 INT8 모델을 같은 CPU·런타임 설정에서 실행한다.

```powershell
.\.venv\Scripts\python.exe evaluation.py checkpoints\onnx_resnet18.onnx --output results\fp32_reference.json
```

평가 도구는 validation 정확도, ONNX 및 external data 파일 크기, batch 1 latency, batch 32 throughput, 별도 프로세스의 최대 RAM을 JSON으로 저장한다. 두 시간 지표는 각각 100회 warm-up 뒤 1,000회 측정하며 원시 시간도 저장한다. CPUExecutionProvider, intra-op 6 threads, inter-op 1 thread, `ORT_ENABLE_ALL` 최적화를 모든 모델에 적용한다. 시간 측정에는 준비된 입력의 `session.run` 호출만 포함한다. 메모리 값에는 Python·ONNX Runtime·모델 초기화와 추론을 포함한 작업 프로세스 전체가 들어간다.

이후 INT8 모델은 같은 장비에서 측정하고 FP32 결과 파일을 참조한다.

```powershell
.\.venv\Scripts\python.exe evaluation.py checkpoints\full_int8.onnx --baseline-result results\fp32_reference.json --output results\full_int8_reference.json
```

인자 없이 `evaluation.py`를 실행하면 FP32 ONNX를 측정하고 `results/generated/`에 시간표시가 붙은 로컬 결과를 저장한다. 이 폴더는 Git에서 제외한다.

## M5 Full INT8 생성과 중간 확인

```powershell
.\.venv\Scripts\python.exe quantize_full_int8.py
.\.venv\Scripts\python.exe inspect_full_int8.py
```

첫 명령은 고정 calibration 1,000장으로 QDQ·MinMax Static PTQ를 적용해 `checkpoints/full_int8.onnx`를 만든다. 두 번째 명령은 M3 매핑의 Conv·Gemm·Add에 대한 Q/DQ·가중치 범위를 확인하고 같은 validation 5,000장에서 FP32와 Full INT8 정확도를 비교한다. 이 정확도 확인은 최종 CPU 성능 측정과 구분한다.

## M6 Weight 양자화 오차 분석

```powershell
.\.venv\Scripts\python.exe analyze_weight_error.py
```

FP32 ONNX와 Full INT8 ONNX의 대응하는 Conv·Gemm weight 21개를 비교한다. INT8 모델에 저장된 scale·zero-point·axis로 weight를 복원하고, weight별 MSE·NMSE와 여섯 그룹별 NMSE를 계산한다. 그룹 NMSE는 각 weight의 제곱 오차 합을 더한 뒤 FP32 weight 제곱 합으로 나눈 값이다. Add에는 weight가 없고 INT32 bias는 분석에서 제외한다.

결과는 `results/generated/m6/`의 CSV, 그래프 PNG, 모델 해시와 계산식을 담은 `metadata.json`에 저장된다. 이 폴더는 Git에서 제외한다. 이 weight 오차는 M7에서 측정할 validation 정확도 민감도와 별개의 지표다.

## M7 그룹별 정확도 민감도

```powershell
.\.venv\Scripts\python.exe quantize_group_int8.py
.\.venv\Scripts\python.exe evaluate_group_sensitivity.py
```

첫 명령은 M3 매핑의 여섯 그룹을 각각 단독으로 양자화한 ONNX 모델을 `checkpoints/`에 생성한다. 여섯 번 모두 동일한 FP32 ONNX, 고정 calibration 1,000장, M5와 같은 QDQ·MinMax·QInt8 설정을 사용한다. 각 모델에서 대상 Conv·Gemm·Add의 Q/DQ 및 비대상 Conv·Gemm의 FP32 weight를 검사하고, 입력 모델·데이터 split·설정·출력 모델 해시를 manifest에 기록한다.

두 번째 명령은 manifest와 그래프를 재검증한 뒤 동일한 validation 5,000장에서 FP32와 그룹별 모델의 정확도를 측정한다. `Sensitivity = FP32 정확도 − 해당 그룹만 양자화한 정확도`이며 단위는 %p이고, 음수도 그대로 기록한다. 정확도 하락이 같으면 공동 순위로 표시한다. `results/generated/m7/`에 순위표 CSV·JSON, 양자화 범위 검증 기록과 그래프를 저장한다. 그룹 경계 Q/DQ의 영향이 정확도에 포함되므로 M6의 weight NMSE 순위와 구분해서 해석한다. Test 데이터는 이 단계에서 평가하지 않는다.
