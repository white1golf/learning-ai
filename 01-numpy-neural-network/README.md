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

## 앞으로 구현할 내용

1. 입력과 가중치의 행렬 연산
2. 활성화 함수
3. 손실 함수
4. 수치 미분과 경사하강법
5. 역전파
6. 2층 신경망 학습

이후 실습도 `02_...py`, `03_...py`처럼 새 파일로 추가합니다.
