# 02. PyTorch로 MNIST 학습하기

01번에서 NumPy로 만든 순전파·손실·역전파·학습 루프를, PyTorch API로 옮겨
실제 이미지 분류(MNIST)까지 확장하는 단계입니다.

전체 학습 순서와 파일 작성 규칙은 상위 폴더의 `README.md`에 기록되어 있습니다.

## 환경 준비

```bash
cd 02-pytorch-mnist
uv sync
uv run python main.py
```

정상적으로 준비되었다면 Python·PyTorch 버전과 작은 행렬 곱 결과가 출력됩니다.
CUDA가 없어도 CPU로 실습할 수 있습니다.

## 실습 기록

| 순서 | 파일 | 학습 내용 | 실행 명령 |
| --- | --- | --- | --- |
| 준비 | `main.py` | PyTorch 환경 확인 | `uv run python main.py` |
| 01 | `01_tensors_and_autograd.py` | 텐서, autograd, Linear와 01번 개념의 대응 | `uv run python 01_tensors_and_autograd.py` |
| 02 | `02_mnist_dataset.py` | MNIST, Dataset, DataLoader, 배치 shape | `uv run python 02_mnist_dataset.py` |
| 03 | `03_training_loop.py` | MLP 분류기, CrossEntropy, 학습/평가 루프 | `uv run python 03_training_loop.py` |
| 04 | `04_overfitting.py` | 과적합 관찰, Dropout과 weight decay 비교 | `uv run python 04_overfitting.py` |

## 현재 학습 흐름

- 01번 폴더에서 XOR까지 직접 학습 루프를 만들었다.
- 02번의 01 파일은 그 지식을 PyTorch 텐서/`loss.backward()`/`nn.Linear`로 옮긴다.
- 02번의 02 파일은 MNIST를 내려받고 Dataset/DataLoader로 배치를 만든다.
- 02번의 03 파일은 모델과 optimizer로 실제 숫자 분류기를 학습·평가한다.
- 02번의 04 파일은 작은 학습셋 + 넓은 모델로 과적합을 보이게 하고 규제로 완화한다.

## 이 폴더의 다음

02번 폴더의 로드맵은 04번에서 한 바퀴 완성된다.  
다음 큰 단계는 상위 `README.md`의 **03. 언어 모델의 출발점 만들기**이다.
