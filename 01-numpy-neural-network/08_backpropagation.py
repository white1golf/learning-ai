"""8단계: 연쇄법칙과 역전파로 여러 매개변수의 기울기를 구한다.

이전 실습과의 연결
------------------
07번에서는 가중치가 하나인 회귀 문제에서 수치 미분으로 기울기를 구하고
경사하강법으로 갱신했다. 그 방법은 맞지만, 매개변수마다 순전파를
여러 번 다시 돌려야 해서 실제 신경망에서는 매우 느리다.

이번 실습에서는 backpropagation(역전파)을 배운다. 역전파는 손실에서
시작해 계산 그래프를 거꾸로 따라가며, 연쇄법칙(chain rule)로 각
매개변수의 기울기를 한 번의 역방향 계산으로 구한다.

    순전파: 입력 -> 은닉층 -> 출력 -> 손실
    역전파: 손실 -> 출력 -> 은닉층 -> 입력 방향의 기울기

문제를 단순하게 만드는 이유
----------------------------
05번처럼 배치와 출력 뉴런이 많으면 수식이 길어져 핵심이 가려진다.
여기서는 샘플 1개, 입력 2개, 은닉 뉴런 2개, 출력 1개인 작은 회귀
문제로 축소한다.

    x  ->  z1 = W1 @ x + b1  ->  a1 = ReLU(z1)
       ->  z2 = W2 @ a1 + b2  ->  loss = (z2 - y)^2

연쇄법칙
--------
f(g(w))처럼 함수가 합성되어 있으면, w에 대한 전체 미분은
각 단계의 지역 미분을 곱해서 구한다.

    dL/dw = (dL/dz) * (dz/dw)

역전파는 이 곱셈을 출력 쪽부터 입력 쪽으로 차례로 적용하는 과정이다.

이번 실습의 다음
----------------
여기서는 기울기를 구하는 방법만 확인한다. 다음 실습에서는 구한
기울기로 2층 신경망의 모든 가중치와 편향을 실제로 학습시킨다.
"""

from __future__ import annotations

import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    """음수는 0으로 만들고 양수는 그대로 통과시킨다."""
    return np.maximum(0.0, values)


def relu_derivative(values: np.ndarray) -> np.ndarray:
    """ReLU의 지역 기울기. 양수면 1, 아니면 0."""
    return (values > 0).astype(float)


def forward(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """작은 2층 회귀 네트워크의 순전파와 중간값을 반환한다."""
    z1 = w1 @ x + b1
    a1 = relu(z1)
    prediction = float(w2 @ a1 + b2)
    return z1, a1, prediction


def mse_loss(prediction: float, target: float) -> float:
    """샘플 하나일 때 제곱 오차."""
    return (prediction - target) ** 2


def numerical_gradient_for_param(
    param: np.ndarray | float,
    loss_fn,
    h: float = 1e-5,
) -> np.ndarray | float:
    """매개변수 하나를 조금씩 바꿔 수치 미분으로 기울기를 근사한다."""
    if np.isscalar(param) or isinstance(param, float):
        left = loss_fn(param - h)
        right = loss_fn(param + h)
        return (right - left) / (2.0 * h)

    grad = np.zeros_like(param, dtype=float)
    original = param.copy()
    it = np.nditer(original, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        index = it.multi_index
        original_value = original[index]

        original[index] = original_value + h
        loss_right = loss_fn(original)

        original[index] = original_value - h
        loss_left = loss_fn(original)

        grad[index] = (loss_right - loss_left) / (2.0 * h)
        original[index] = original_value
        it.iternext()
    return grad


def explain_why_backprop() -> None:
    """수치 미분이 느린 이유를 매개변수 개수로 설명한다."""
    print("1. 수치 미분만으로 학습하면 느린 이유")
    print("   매개변수 하나마다 손실을 좌우 두 번 계산해야 한다.")
    print("   예: 가중치·편향이 100개면 기울기만 구해도 순전파 약 200번.")
    print("   역전파는 순전파 1번 + 역방향 계산 1번으로 모든 기울기를 구한다.\n")


def explain_chain_rule_scalar() -> None:
    """스칼라 합성함수로 연쇄법칙 아이디어를 먼저 확인한다."""
    # loss = (w * x - y)^2 , x=2, y=8, w=1
    x = 2.0
    y = 8.0
    w = 1.0

    prediction = w * x
    error = prediction - y
    loss = error**2

    dloss_derror = 2.0 * error
    derror_dprediction = 1.0
    dprediction_dw = x
    dloss_dw = dloss_derror * derror_dprediction * dprediction_dw

    print("2. 연쇄법칙 손계산 (가중치 하나)")
    print(f"   prediction = w × x = {w:.1f} × {x:.1f} = {prediction:.1f}")
    print(f"   loss = (prediction - y)² = ({prediction:.1f} - {y:.1f})² = {loss:.1f}")
    print("   dL/dw = (dL/derror) × (derror/dpred) × (dpred/dw)")
    print(
        f"         = ({dloss_derror:.1f}) × ({derror_dprediction:.1f}) "
        f"× ({dprediction_dw:.1f}) = {dloss_dw:.1f}"
    )
    print("   07번의 수치 미분과 같은 방향·크기의 기울기를 수식으로 얻었다.\n")


def create_example_values() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """관찰하기 쉬운 작은 네트워크의 예제 값을 만든다."""
    x = np.array([1.0, -2.0])
    w1 = np.array(
        [
            [0.5, -0.3],
            [-0.2, 0.4],
        ]
    )
    b1 = np.array([0.1, -0.1])
    w2 = np.array([0.6, -0.5])
    b2 = 0.2
    target = 1.0
    return x, w1, b1, w2, b2, target


def explain_forward_and_backprop() -> dict[str, np.ndarray | float]:
    """순전파 중간값을 저장한 뒤, 역전파로 기울기를 단계별로 구한다."""
    x, w1, b1, w2, b2, target = create_example_values()

    z1 = w1 @ x + b1
    a1 = relu(z1)
    prediction = float(w2 @ a1 + b2)
    error = prediction - target
    loss = error**2

    print("3. 작은 2층 네트워크의 순전파")
    print(f"   입력 x: {x}")
    print(f"   z1 = W1 @ x + b1 = {z1}")
    print(f"   a1 = ReLU(z1)    = {a1}")
    print(f"   prediction = W2 @ a1 + b2 = {prediction:.4f}")
    print(f"   target = {target:.1f}")
    print(f"   loss = (prediction - target)² = {loss:.4f}\n")

    # 역전파: 출력에서 입력 방향으로
    dloss_dpred = 2.0 * error
    dpred_dw2 = a1
    dpred_db2 = 1.0
    dw2 = dloss_dpred * dpred_dw2
    db2 = dloss_dpred * dpred_db2

    dpred_da1 = w2
    da1 = dloss_dpred * dpred_da1
    dz1 = da1 * relu_derivative(z1)

    dw1 = np.outer(dz1, x)
    db1 = dz1

    print("4. 역전파: 손실에서 시작해 층마다 기울기를 전달")
    print(f"   dL/dprediction = 2 × error = {dloss_dpred:.4f}")
    print(f"   dW2 = dL/dpred × a1        = {dw2}")
    print(f"   db2 = dL/dpred × 1         = {db2:.4f}")
    print(f"   da1 = dL/dpred × W2        = {da1}")
    print(f"   dz1 = da1 × ReLU'(z1)      = {dz1}")
    print("   dW1 = outer(dz1, x)")
    print(f"       = \n{dw1}")
    print(f"   db1 = dz1                  = {db1}")
    print("   ReLU'는 음수 경로의 기울기를 0으로 막아 학습이 그쪽으로 흐르지 않게 한다.\n")

    return {
        "x": x,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "target": target,
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
        "loss": loss,
    }


def compare_with_numerical_gradients(grads: dict[str, np.ndarray | float]) -> None:
    """해석적 역전파 결과와 수치 미분이 같은지 확인한다."""
    x = grads["x"]
    w1 = grads["w1"].copy()
    b1 = grads["b1"].copy()
    w2 = grads["w2"].copy()
    b2 = float(grads["b2"])
    target = float(grads["target"])

    def loss_with_w1(candidate_w1: np.ndarray) -> float:
        _, _, prediction = forward(x, candidate_w1, b1, w2, b2)
        return mse_loss(prediction, target)

    def loss_with_b1(candidate_b1: np.ndarray) -> float:
        _, _, prediction = forward(x, w1, candidate_b1, w2, b2)
        return mse_loss(prediction, target)

    def loss_with_w2(candidate_w2: np.ndarray) -> float:
        _, _, prediction = forward(x, w1, b1, candidate_w2, b2)
        return mse_loss(prediction, target)

    def loss_with_b2(candidate_b2: float) -> float:
        _, _, prediction = forward(x, w1, b1, w2, candidate_b2)
        return mse_loss(prediction, target)

    num_dw1 = numerical_gradient_for_param(w1, loss_with_w1)
    num_db1 = numerical_gradient_for_param(b1, loss_with_b1)
    num_dw2 = numerical_gradient_for_param(w2, loss_with_w2)
    num_db2 = numerical_gradient_for_param(b2, loss_with_b2)

    print("5. 역전파 기울기 vs 수치 미분 기울기")
    print(f"   dW2 역전파: {grads['dw2']}")
    print(f"   dW2 수치:   {num_dw2}")
    print(f"   db2 역전파: {grads['db2']:.6f}, 수치: {num_db2:.6f}")
    print(f"   db1 역전파: {grads['db1']}")
    print(f"   db1 수치:   {num_db1}")
    print("   dW1 역전파:")
    print(grads["dw1"])
    print("   dW1 수치:")
    print(num_dw1)
    print(
        "   거의 같으면 역전파 구현이 맞다. "
        f"일치 여부: {np.allclose(grads['dw1'], num_dw1) and np.allclose(grads['dw2'], num_dw2)}\n"
    )


def explain_one_parameter_update(grads: dict[str, np.ndarray | float]) -> None:
    """구한 기울기로 매개변수 하나를 갱신해 손실이 줄는지 본다."""
    x = grads["x"]
    w1 = grads["w1"].copy()
    b1 = grads["b1"].copy()
    w2 = grads["w2"].copy()
    b2 = float(grads["b2"])
    target = float(grads["target"])
    learning_rate = 0.05

    old_loss = float(grads["loss"])
    w2 = w2 - learning_rate * grads["dw2"]
    b2 = b2 - learning_rate * float(grads["db2"])
    w1 = w1 - learning_rate * grads["dw1"]
    b1 = b1 - learning_rate * grads["db1"]

    _, _, new_prediction = forward(x, w1, b1, w2, b2)
    new_loss = mse_loss(new_prediction, target)

    print("6. 역전파 기울기로 한 번 갱신")
    print(f"   학습률: {learning_rate}")
    print(f"   갱신 후 예측: {new_prediction:.4f}")
    print(f"   손실 변화: {old_loss:.4f} -> {new_loss:.4f}")
    print("   모든 매개변수를 한 번에 갱신해도 손실이 줄어든다.\n")


def explain_next_step() -> None:
    """다음 실습에서 전체 학습 루프로 확장할 것을 안내한다."""
    print("7. 다음 실습")
    print("   지금까지는 기울기를 '구하는 방법'을 검증했다.")
    print("   다음에는 배치 데이터와 학습 루프로 2층 신경망을 실제로 학습한다.")


def main() -> None:
    explain_why_backprop()
    explain_chain_rule_scalar()
    grads = explain_forward_and_backprop()
    compare_with_numerical_gradients(grads)
    explain_one_parameter_update(grads)
    explain_next_step()


if __name__ == "__main__":
    main()
