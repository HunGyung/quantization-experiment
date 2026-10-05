# CIFAR-10 ResNet-18 INT8 PTQ 실험

FP32 ResNet-18을 기준으로 ONNX Runtime CPU 환경에서 Static PTQ의 정확도와 효율을 비교하고, ResNet stage별 양자화 민감도를 분석한다.

## Python 환경

Windows와 Python 3.14.4에서 검증한 패키지 버전은 [requirements.txt](requirements.txt)에 고정했다. 새 가상환경을 만든 뒤 PyTorch·torchvision wheel을 먼저 설치하고 나머지 패키지를 설치한다. NVIDIA GPU를 사용하는 경우, 이번 실험 환경의 CUDA 13.0 wheel 설치 명령은 다음과 같다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CPU만 사용하는 장비에서는 위 PyTorch 설치 줄 대신 `--index-url https://download.pytorch.org/whl/cpu`를 사용한다. `requirements.txt`는 PyTorch의 공개 버전을 지정하므로 먼저 설치한 CUDA·CPU wheel을 그대로 사용할 수 있다. 양자화 및 성능 비교는 아래의 ONNX Runtime CPU 설정으로 실행한다.

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

## M8 Selective INT8 후보 비교

M7에서 Layer 1과 Layer 3이 공동 최고 민감도였으므로, Layer 1만 FP32 유지, Layer 3만 FP32 유지, 두 그룹 모두 FP32 유지의 세 후보를 미리 정했다. 각 후보는 같은 FP32 ONNX에서 출발해 나머지 M3 Conv·Gemm·Add 노드만 M5와 동일한 조건으로 양자화한다.

```powershell
.\.venv\Scripts\python.exe quantize_selective_int8.py
.\.venv\Scripts\python.exe evaluate_selective_int8.py
```

첫 명령은 후보 ONNX 3개와 해시·설정 manifest를 `checkpoints/`에 만든다. 두 번째 명령은 M7/M8 manifest와 실제 그래프를 확인하고, FP32·Full INT8·후보 3개의 validation 정확도와 파일 크기를 미리 비교한다. 이 결과만으로 최종 모델을 정하지 않는다.

코드와 모델을 준비한 뒤 성능을 비교할 장비 하나에서 아래 명령을 실행한다. FP32부터 다섯 모델을 모두 같은 CPU·ORT 설정과 M4 프로토콜로 새로 측정하고 원시 결과와 비교표를 `results/generated/m8/`에 저장한다. 다른 장비나 과거 smoke 결과와 섞지 않는다.

```powershell
.\.venv\Scripts\python.exe evaluate_selective_int8.py --benchmark
```

최종 선택 규칙은 **세 Selective 후보 중 validation 정답 수가 가장 많은 모델**을 우선하고, 동률이면 batch 1 latency 중앙값이 낮은 순서, batch 32 throughput이 높은 순서, 총 모델 크기가 작은 순서, peak RAM이 낮은 순서로 결정한다. 완전히 같으면 위에 적은 후보 순서를 따른다. 정확도와 효율의 차이를 비교표에서 확인한 다음 아래 명령으로 저장된 benchmark를 재검증하고 `checkpoints/selective_int8.onnx`와 설정 파일을 동결한다. 이미 동결된 파일은 덮어쓰지 않는다. Test는 M9에서만 평가한다.

```powershell
.\.venv\Scripts\python.exe evaluate_selective_int8.py --freeze
```

## M9 최종 Test 정확도

M8 benchmark와 동결을 마친 뒤, 같은 데스크탑에서 아래 명령을 **한 번** 실행한다. 실행 전 동결 설정, M8 비교 기록, 데이터 split, 세 ONNX 모델의 해시를 확인한다. Test 10,000장에서 FP32·Full INT8·동결된 Selective INT8의 Top-1 정확도를 측정하고 `results/generated/m9/test_comparison.json`과 `.csv`를 저장한다.

```powershell
.\.venv\Scripts\python.exe evaluate_test.py
```

CSV의 latency·throughput·RAM은 test 이미지로 다시 측정한 값이 아니라 같은 데스크탑의 M8 benchmark에서 가져온 값이다. Test 결과를 보고 Selective 후보를 다시 고르지 않는다. 최종 결과 문서에는 M6·M7 분석과 이 비교표, 실행 환경과 원시 기록을 함께 정리한다.
