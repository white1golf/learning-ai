"""9단계: 역전파와 경사하강법으로 2층 신경망을 학습한다.

이전 실습과의 연결
------------------
08번에서는 샘플 하나에서 역전파 기울기를 구하고, 수치 미분과 비교해
구현이 맞는지 확인했다. 갱신도 한 번만 해 보았다.

이번 실습에서는 그 과정을 반복하는 training loop(학습 루프)를 만든다.

    1. 순전파로 예측을 계산한다.
    2. 손실을 계산한다.
    3. 역전파로 모든 매개변수의 기울기를 구한다.
    4. 경사하강법으로 매개변수를 갱신한다.
    5. 1~4를 여러 epoch(에폭) 동안 반복한다.

왜 XOR인가
-----------
입력이 두 개이고 정답이 0 또는 1인 XOR 문제는, 직선 하나로 나누기
어렵다. 은닉층과 ReLU가 있는 2층 네트워크는 이런 비선형 경계를
학습할 수 있다. 05~08에서 만든 구성 요소를 한 줄로 이어 실제로
학습이 되는지 관찰하기 좋은 작은 예제다.

    입력 (x0, x1) -> 정답
    (0, 0) -> 0
    (0, 1) -> 1
    (1, 0) -> 1
    (1, 1) -> 0

배치 표기
---------
05번과 같이 행이 샘플인 배치 표기를 사용한다.

    X.shape  == (4, 2)
    W1.shape == (4, 2)   # 은닉 뉴런 4개, 입력 특성 2개
    W2.shape == (1, 4)   # 출력 뉴런 1개
    Y.shape  == (4, 1)

    Z1 = X @ W1.T + b1
    A1 = ReLU(Z1)
    Z2 = A1 @ W2.T + b2
    loss = mean((Z2 - Y)^2)

이번 단계로 01번 폴더의 핵심 흐름
(순전파 -> 손실 -> 역전파 -> 갱신)이 한 바퀴 완성된다.
다음 큰 단계는 PyTorch로 MNIST를 학습하는 02번이다.
"""

from __future__ import annotations

import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    """음수는 0으로 만들고 양수는 그대로 통과시킨다."""
    return np.maximum(0.0, values)


def relu_derivative(values: np.ndarray) -> np.ndarray:
    """ReLU의 지역 기울기. 양수면 1, 아니면 0."""
    return (values > 0).astype(float)


def create_xor_data() -> tuple[np.ndarray, np.ndarray]:
    """XOR 진리표를 배치 입력과 정답으로 만든다."""
    inputs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    targets = np.array(
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ]
    )
    return inputs, targets


def initialize_parameters(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """작은 난수로 2층 네트워크 매개변수를 초기화한다."""
    w1 = rng.normal(loc=0.0, scale=0.5, size=(4, 2))
    b1 = np.zeros(4)
    w2 = rng.normal(loc=0.0, scale=0.5, size=(1, 4))
    b2 = np.zeros(1)
    return w1, b1, w2, b2


def forward(
    inputs: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """배치 순전파. 역전파에 쓸 중간값도 함께 반환한다."""
    z1 = inputs @ w1.T + b1
    a1 = relu(z1)
    z2 = a1 @ w2.T + b2
    return z1, a1, z2


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """배치 평균 제곱 오차."""
    return float(np.mean((predictions - targets) ** 2))


def backward(
    inputs: np.ndarray,
    targets: np.ndarray,
    z1: np.ndarray,
    a1: np.ndarray,
    predictions: np.ndarray,
    w2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """배치 평균 손실에 대한 역전파 기울기를 계산한다."""
    batch_size = inputs.shape[0]
    dloss_dpred = 2.0 * (predictions - targets) / batch_size

    dw2 = dloss_dpred.T @ a1
    db2 = np.sum(dloss_dpred, axis=0)

    dloss_da1 = dloss_dpred @ w2
    dloss_dz1 = dloss_da1 * relu_derivative(z1)

    dw1 = dloss_dz1.T @ inputs
    db1 = np.sum(dloss_dz1, axis=0)
    return dw1, db1, dw2, db2


def update_parameters(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    dw1: np.ndarray,
    db1: np.ndarray,
    dw2: np.ndarray,
    db2: np.ndarray,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """경사하강법으로 매개변수를 한 번 갱신한다."""
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    return w1, b1, w2, b2


def explain_problem() -> None:
    """학습 문제와 전체 루프를 설명한다."""
    print("1. 이번 학습 문제: XOR")
    print("   한 번의 선형 분류로는 나누기 어려운 네 점이다.")
    print("   2층 네트워크가 손실을 줄이며 정답 패턴을 익히는지 본다.\n")

    print("2. 학습 루프")
    print("   epoch마다: 순전파 -> MSE 손실 -> 역전파 -> 매개변수 갱신")
    print("   epoch(에폭)은 전체 학습 데이터를 한 바퀴 도는 단위다.\n")


def show_predictions(
    title: str,
    inputs: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    loss: float,
) -> None:
    """입력, 정답, 예측을 표처럼 출력한다."""
    print(title)
    print(f"   손실(MSE): {loss:.6f}")
    print("   x0  x1 | 정답 | 예측")
    print("   -------+------+--------")
    for (x0, x1), target, prediction in zip(inputs, targets, predictions):
        print(
            f"   {x0:.0f}   {x1:.0f}  |  {target[0]:.0f}   | "
            f"{prediction[0]:.4f}"
        )
    print()


def train_xor_network() -> None:
    """XOR 데이터로 2층 네트워크를 학습한다."""
    rng = np.random.default_rng(seed=0)
    inputs, targets = create_xor_data()
    w1, b1, w2, b2 = initialize_parameters(rng)

    learning_rate = 0.1
    epoch_count = 2000
    log_every = 400

    z1, a1, predictions = forward(inputs, w1, b1, w2, b2)
    initial_loss = mse_loss(predictions, targets)
    show_predictions("3. 학습 전 예측", inputs, targets, predictions, initial_loss)

    print("4. 학습 중 손실 변화")
    print("   epoch | 손실")
    print("   ------+----------")
    print(f"   {0:>5} | {initial_loss:.6f}")

    for epoch in range(1, epoch_count + 1):
        z1, a1, predictions = forward(inputs, w1, b1, w2, b2)
        loss = mse_loss(predictions, targets)
        dw1, db1, dw2, db2 = backward(inputs, targets, z1, a1, predictions, w2)
        w1, b1, w2, b2 = update_parameters(
            w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate
        )

        if epoch % log_every == 0:
            print(f"   {epoch:>5} | {loss:.6f}")

    z1, a1, predictions = forward(inputs, w1, b1, w2, b2)
    final_loss = mse_loss(predictions, targets)
    print()
    show_predictions("5. 학습 후 예측", inputs, targets, predictions, final_loss)

    rounded = np.round(predictions)
    correct = np.sum(rounded == targets)
    print("6. 관찰 포인트")
    print(f"   반올림 후 맞춘 개수: {correct}/{len(targets)}")
    print("   손실이 줄어들고 예측이 정답 0/1에 가까워지면 학습이 된 것이다.")
    print("   08번에서 검증한 역전파가, 반복되면 실제 학습 엔진이 된다.\n")


def explain_next_stage() -> None:
    """01번 폴더 완료와 다음 큰 단계를 안내한다."""
    print("7. 다음 큰 단계")
    print("   01번에서 NumPy로 순전파·손실·역전파·학습 루프를 직접 만들었다.")
    print("   다음에는 PyTorch로 MNIST 숫자 분류기를 학습한다.")
    print("   그때 autograd가 이번 08~09번의 역전파를 대신 수행한다.")


def main() -> None:
    explain_problem()
    train_xor_network()
    explain_next_stage()


if __name__ == "__main__":
    main()
