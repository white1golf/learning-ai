# 01. NumPy로 신경망 만들기

딥러닝 프레임워크 없이 NumPy만 사용해 작은 신경망을 직접 구현하는 첫 번째 실습입니다.

전체 학습 순서와 파일 작성 규칙은 상위 폴더의 `README.md`에 기록되어 있습니다.

## 환경 준비

별도로 가상 환경을 활성화할 필요 없이 다음 명령을 실행합니다.

```bash
uv sync
uv run python main.py
```

정상적으로 준비되었다면 Python과 NumPy 버전, 행렬 연산 결과가 출력됩니다.

## 실습 기록

각 학습 단계는 이전 내용을 덮어쓰지 않고 번호가 붙은 별도 파일로 보존합니다.

| 순서 | 파일 | 학습 내용 | 실행 명령 |
| --- | --- | --- | --- |
| 준비 | `main.py` | Python, NumPy 및 행렬 연산 확인 | `uv run python main.py` |
| 01 | `01_weights_and_bias.py` | 단일 뉴런의 입력, 가중치, 편향 | `uv run python 01_weights_and_bias.py` |
| 02 | `02_dense_layer.py` | 여러 뉴런, 완전연결층과 배열 shape | `uv run python 02_dense_layer.py` |
| 03 | `03_batch.py` | 여러 입력 샘플의 배치 처리, 전치와 브로드캐스팅 | `uv run python 03_batch.py` |
| 04 | `04_activation_functions.py` | Step, Sigmoid, ReLU와 비선형성 | `uv run python 04_activation_functions.py` |
| 05 | `05_two_layer_forward.py` | 두 개의 완전연결층과 ReLU를 연결한 순전파 | `uv run python 05_two_layer_forward.py` |
| 06 | `06_loss_functions.py` | 회귀와 분류, 클래스, MSE, Softmax와 교차 엔트로피 | `uv run python 06_loss_functions.py` |
| 07 | `07_numerical_gradient_descent.py` | 수치 미분, 기울기와 경사하강법으로 가중치 갱신 | `uv run python 07_numerical_gradient_descent.py` |
| 08 | `08_backpropagation.py` | 연쇄법칙과 역전파로 여러 매개변수의 기울기 계산 | `uv run python 08_backpropagation.py` |
| 09 | `09_two_layer_training.py` | 역전파와 학습 루프로 2층 네트워크가 XOR 학습 | `uv run python 09_two_layer_training.py` |

## 현재 학습 흐름

- 05번에서는 사람이 임의로 정한 weights(가중치)와 biases(편향)로 forward pass(순전파)를 계산했다. 아직 학습은 일어나지 않았다.
- 06번에서는 regression(회귀)과 classification(분류)의 차이를 먼저 익힌다. MSE는 회귀용 별도 예제로 살펴보고, 05번의 출력은 두 class(클래스)의 점수라고 가정해 분류 손실을 계산한다.
- 07번에서는 가중치가 하나인 회귀 문제로 잠시 단순화한다. 매개변수 변화에 따른 손실 변화를 수치 미분으로 구하고, 경사하강법으로 가중치를 반복 갱신한다.
- 08번에서는 작은 2층 네트워크에서 연쇄법칙(역전파)으로 모든 기울기를 한 번에 구하고, 수치 미분과 비교해 검증한다.
- 09번에서는 배치 XOR 데이터로 순전파·손실·역전파·갱신을 반복해 2층 신경망을 실제로 학습한다.

## 이 폴더의 다음

01번 폴더의 핵심 흐름은 09번에서 한 바퀴 완성되었다.  
다음 큰 단계는 `02-pytorch-mnist/` 이다.
