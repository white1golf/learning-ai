"""1단계: NumPy 신경망 지식을 PyTorch 텐서와 autograd로 연결한다.

이전 단계와의 연결
------------------
01번 폴더에서는 NumPy로 다음을 직접 구현했다.

    순전파 -> 손실 -> 역전파(연쇄법칙) -> 매개변수 갱신

그때 손으로 작성했던 dW, db 계산을 PyTorch는 autograd가 대신한다.
이번 실습은 MNIST 분류기로 들어가기 전에, 그 대응 관계를 먼저 확인한다.

텐서(tensor)
------------
tensor는 NumPy ndarray와 비슷하지만, GPU로 옮길 수 있고
requires_grad=True이면 계산 과정을 기록해 자동 미분할 수 있다.

    NumPy:  values = np.array([1.0, 2.0])
    PyTorch: values = torch.tensor([1.0, 2.0])

autograd
--------
01번의 08~09에서 한 일:

    1. 순전파 중간값을 저장한다.
    2. 손실에서 시작해 dL/d(각 값)을 뒤로 전달한다.
    3. 구한 기울기로 매개변수를 갱신한다.

PyTorch에서는:

    prediction = ...
    loss = ...
    loss.backward()   # 2번을 자동으로 수행
    # w.grad 에 dL/dw 가 들어 있다

이번 실습의 다음
----------------
텐서와 autograd를 이해한 뒤, MNIST 데이터셋과 DataLoader로 넘어간다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compare_numpy_style_and_tensor() -> None:
    """NumPy에서 쓰던 행렬 곱이 텐서에서도 같은 형태임을 본다."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    w = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    # 01번의 dense: inputs @ weights.T
    y = x @ w.T

    print("1. 텐서는 NumPy 배열과 비슷하다")
    print(f"   x.shape = {tuple(x.shape)}, w.shape = {tuple(w.shape)}")
    print(f"   y = x @ w.T = {y}")
    print("   아직 requires_grad가 없으면 자동 미분 대상이 아니다.\n")


def explain_autograd_on_one_weight() -> None:
    """08번 스칼라 예제를 autograd로 다시 계산한다."""
    # prediction = w * x, loss = (prediction - y)^2
    # 08번 손계산: dL/dw = -24
    x = torch.tensor(2.0)
    y = torch.tensor(8.0)
    w = torch.tensor(1.0, requires_grad=True)

    prediction = w * x
    loss = (prediction - y) ** 2
    loss.backward()

    print("2. 08번 스칼라 예제를 autograd로")
    print(f"   w={w.item():.1f}, x={x.item():.1f}, y={y.item():.1f}")
    print(f"   prediction={prediction.item():.1f}, loss={loss.item():.1f}")
    print(f"   w.grad = {w.grad.item():.1f}  (손계산 dL/dw = -24와 같아야 한다)")
    print("   loss.backward()가 연쇄법칙 역전파를 대신 수행했다.\n")


def train_one_weight_with_autograd() -> None:
    """07~09처럼 기울기 반대 방향으로 가중치 하나를 여러 번 갱신한다."""
    x = torch.tensor(2.0)
    y = torch.tensor(8.0)
    w = torch.tensor(1.0, requires_grad=True)
    learning_rate = 0.05
    steps = 10

    print("3. autograd + 경사하강법으로 가중치 하나 학습")
    print("   단계 | 가중치  | 예측값  | 손실")
    print("   -----+---------+---------+----------")

    for step in range(steps + 1):
        prediction = w * x
        loss = (prediction - y) ** 2
        print(
            f"   {step:>4} | {w.item():>7.4f} | "
            f"{prediction.item():>7.4f} | {loss.item():>8.6f}"
        )

        if step == steps:
            break

        loss.backward()
        with torch.no_grad():
            w -= learning_rate * w.grad
            w.grad.zero_()

    print("   w는 4에, prediction은 8에 가까워진다.")
    print("   torch.no_grad() 구간은 '갱신 계산을 미분 기록에서 제외'한다는 뜻이다.")
    print("   w.grad.zero_()는 다음 backward 전에 기울기를 비운다.\n")


def compare_manual_dense_and_linear() -> None:
    """09번의 Dense 계산과 nn.Linear가 같은 역할임을 확인한다."""
    torch.manual_seed(0)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    manual = nn.Linear(in_features=2, out_features=3, bias=True)
    y_manual_api = manual(x)

    # Linear는 내부적으로 y = x @ W.T + b 를 수행한다.
    y_from_math = x @ manual.weight.T + manual.bias

    print("4. nn.Linear는 01번의 Dense Layer에 해당한다")
    print(f"   입력 shape: {tuple(x.shape)}")
    print(f"   weight shape: {tuple(manual.weight.shape)}  # (out, in)")
    print(f"   bias shape: {tuple(manual.bias.shape)}")
    print(f"   Linear 출력: {y_manual_api}")
    print(f"   수식 출력:   {y_from_math}")
    print(f"   같은가? {torch.allclose(y_manual_api, y_from_math)}\n")


def build_tiny_mlp_like_xor_net() -> None:
    """09번 XOR 네트워크를 nn.Sequential로 표현해 본다."""
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = model(x)

    print("5. 09번과 같은 구조의 작은 MLP")
    print("   Linear(2->4) -> ReLU -> Linear(4->1)")
    print(f"   XOR 입력 4개에 대한 초기 출력:\n{y}")
    print("   아직 학습 전이라 값은 의미 없다.")
    print("   다만 구조를 PyTorch API로 옮긴 형태임을 보면 된다.\n")


def explain_next_step() -> None:
    """다음 실습에서 데이터셋으로 확장할 것을 안내한다."""
    print("6. 다음 실습")
    print("   지금까지: 텐서, autograd, Linear가 01번 개념과 어떻게 대응하는지")
    print("   다음: MNIST 손글씨 숫자 데이터셋과 DataLoader")


def main() -> None:
    compare_numpy_style_and_tensor()
    explain_autograd_on_one_weight()
    train_one_weight_with_autograd()
    compare_manual_dense_and_linear()
    build_tiny_mlp_like_xor_net()
    explain_next_step()


if __name__ == "__main__":
    main()
