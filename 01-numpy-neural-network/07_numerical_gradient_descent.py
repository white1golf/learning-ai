"""7단계: 수치 미분으로 기울기를 구하고 가중치를 갱신한다.

이전 실습과의 연결
------------------
05번에서는 사람이 임의로 정한 weights(가중치)와 biases(편향)로
forward pass(순전파)를 수행했다. 06번에서는 그 예측과 정답의 차이를
loss(손실)로 측정했다. 그러나 손실 값만 보고는 어떤 매개변수를 어느
방향으로 바꿔야 손실이 줄어드는지 알 수 없다.

이번에는 weight를 아주 조금 바꿨을 때 손실이 얼마나 변하는지 측정한다.
이 변화의 비율을 derivative(미분값) 또는 slope(기울기)라고 한다.

    기울기 > 0 -> weight가 커질수록 손실도 커지는 중
    기울기 < 0 -> weight가 커질수록 손실은 작아지는 중
    기울기 = 0 -> 현재 위치 주변에서 손실 변화가 거의 없음

문제를 단순하게 만드는 이유
----------------------------
2층 신경망에는 여러 가중치와 편향이 있어 모든 값을 한꺼번에 보면
학습의 핵심을 관찰하기 어렵다. 이번 실습에서는 입력 1개, 가중치 1개,
정답 1개만 사용하는 아주 작은 회귀 문제로 잠시 축소한다.

    input = 2
    target = 8
    prediction = input * weight
    loss = (prediction - target) ** 2

bias는 기울기로 가중치를 갱신하는 과정에만 집중하기 위해 생략한다.
weight가 4가 되면 prediction은 2 * 4 = 8이 되어 정답과 같아진다.

수치 미분
---------
numerical differentiation(수치 미분)은 현재 weight의 양쪽 값을 조금씩
확인해 기울기를 근사한다. 여기서 h는 아주 작은 변화량이다.

                 loss(weight + h) - loss(weight - h)
    gradient = ---------------------------------------
                                  2h

매개변수가 하나일 때는 derivative(미분값)라고 부를 수 있다. 신경망처럼
매개변수가 여러 개라면 각 매개변수의 미분값을 모은 것을 gradient(그래디언트,
기울기)라고 한다.

경사하강법
---------
gradient descent(경사하강법)는 기울기의 반대 방향으로 매개변수를 움직여
손실을 줄이는 방법이다.

    new_weight = weight - learning_rate * gradient

learning rate(학습률)는 한 번에 얼마나 움직일지 정한다. 너무 크면 손실이
가장 작은 지점을 지나칠 수 있고, 너무 작으면 학습에 많은 반복이 필요하다.

이번 실습의 다음
----------------
이 방법을 2층 신경망에 적용하려면 모든 가중치와 편향을 하나씩 바꾸며
매번 전체 순전파와 손실 계산을 반복해야 한다. 가능하지만 매우 느리다.
다음 실습에서는 backpropagation(역전파)으로 모든 기울기를 효율적으로
계산하는 원리를 배운다.
"""


def predict(input_value: float, weight: float) -> float:
    """입력과 가중치 하나로 예측값을 만든다."""
    return input_value * weight


def mse_for_one_sample(
    weight: float,
    input_value: float,
    target: float,
) -> float:
    """가중치 하나가 만든 예측의 제곱 오차를 반환한다.

    샘플이 하나뿐이므로 제곱 오차 하나의 평균은 그 값 자체다.
    """
    prediction = predict(input_value, weight)
    error = prediction - target
    return error**2


def numerical_gradient(
    weight: float,
    input_value: float,
    target: float,
    h: float = 0.0001,
) -> float:
    """중앙 차분으로 weight에 대한 손실의 기울기를 근사한다."""
    loss_left = mse_for_one_sample(weight - h, input_value, target)
    loss_right = mse_for_one_sample(weight + h, input_value, target)
    return (loss_right - loss_left) / (2.0 * h)


def explain_current_position() -> None:
    """06번의 손실 계산이 매개변수 학습으로 이어지는 위치를 설명한다."""
    print("1. 손실을 계산한 다음 필요한 것")
    print("   05번: 임의의 매개변수로 예측한다.")
    print("   06번: 예측과 정답의 차이를 손실로 측정한다.")
    print("   07번: 손실을 줄이는 방향을 찾아 가중치를 갱신한다.")
    print("   이번에는 원리를 보기 위해 가중치가 하나인 문제로 축소한다.\n")


def explain_numerical_gradient() -> float:
    """현재 가중치 양쪽의 손실로 수치 미분 과정을 풀어본다."""
    input_value = 2.0
    target = 8.0
    weight = 1.0
    h = 0.0001

    current_prediction = predict(input_value, weight)
    current_loss = mse_for_one_sample(weight, input_value, target)
    left_weight = weight - h
    right_weight = weight + h
    left_loss = mse_for_one_sample(left_weight, input_value, target)
    right_loss = mse_for_one_sample(right_weight, input_value, target)
    gradient = numerical_gradient(weight, input_value, target, h)

    print("2. 가중치 하나로 만든 회귀 문제")
    print(f"   입력: {input_value:.1f}, 정답: {target:.1f}")
    print(f"   현재 가중치: {weight:.1f}")
    print(
        f"   예측: input × weight = {input_value:.1f} × {weight:.1f} "
        f"= {current_prediction:.1f}"
    )
    print(
        f"   손실: (prediction - target)² = "
        f"({current_prediction:.1f} - {target:.1f})² = {current_loss:.1f}\n"
    )

    print("3. 수치 미분으로 손실의 기울기 근사")
    print(f"   작은 변화량 h: {h}")
    print(f"   weight - h = {left_weight:.4f}, 손실 = {left_loss:.8f}")
    print(f"   weight + h = {right_weight:.4f}, 손실 = {right_loss:.8f}")
    print("   기울기 = (오른쪽 손실 - 왼쪽 손실) / (2 × h)")
    print(
        f"           = ({right_loss:.8f} - {left_loss:.8f}) "
        f"/ {2.0 * h:.4f}"
    )
    print(f"           = {gradient:.4f}")
    print("   기울기가 음수이므로 weight를 키우면 손실이 작아진다.\n")
    return gradient


def explain_one_update(gradient: float) -> None:
    """기울기와 학습률로 가중치를 한 번 갱신한다."""
    input_value = 2.0
    target = 8.0
    weight = 1.0
    learning_rate = 0.05

    old_loss = mse_for_one_sample(weight, input_value, target)
    new_weight = weight - learning_rate * gradient
    new_prediction = predict(input_value, new_weight)
    new_loss = mse_for_one_sample(new_weight, input_value, target)

    print("4. 경사하강법으로 가중치 한 번 갱신")
    print(f"   학습률: {learning_rate}")
    print("   new_weight = weight - learning_rate × gradient")
    print(
        f"              = {weight:.1f} - {learning_rate:.2f} "
        f"× ({gradient:.1f})"
    )
    print(f"              = {new_weight:.1f}")
    print(f"   새로운 예측: {input_value:.1f} × {new_weight:.1f} = {new_prediction:.1f}")
    print(f"   손실 변화: {old_loss:.2f} -> {new_loss:.2f}")
    print("   가중치를 한 번 바꾸자 예측은 정답에 가까워지고 손실은 줄었다.\n")


def train_single_weight() -> None:
    """수치 미분과 경사하강법을 반복해 가중치 하나를 학습한다."""
    input_value = 2.0
    target = 8.0
    weight = 1.0
    learning_rate = 0.05
    update_count = 10

    print("5. 같은 과정을 반복하면 학습이 된다")
    print("   단계 | 가중치  | 예측값  | 손실")
    print("   -----+---------+---------+----------")

    for step in range(update_count + 1):
        prediction = predict(input_value, weight)
        loss = mse_for_one_sample(weight, input_value, target)
        print(f"   {step:>4} | {weight:>7.4f} | {prediction:>7.4f} | {loss:>8.6f}")

        if step < update_count:
            gradient = numerical_gradient(weight, input_value, target)
            weight = weight - learning_rate * gradient

    print("\n   weight는 4에, prediction은 정답 8에 가까워진다.")
    print("   동시에 loss는 0에 가까워진다. 이것이 가장 작은 형태의 학습이다.\n")


def explain_next_step() -> None:
    """가중치 하나의 학습과 2층 신경망의 역전파를 연결한다."""
    print("6. 다시 2층 신경망으로 확장하면")
    print("   실제 신경망은 모든 weight와 bias마다 기울기가 필요하다.")
    print("   수치 미분은 매개변수 하나마다 손실을 여러 번 계산해야 해서 느리다.")
    print("   다음에는 역전파로 모든 기울기를 한 번에 효율적으로 계산한다.")


def main() -> None:
    explain_current_position()
    gradient = explain_numerical_gradient()
    explain_one_update(gradient)
    train_single_weight()
    explain_next_step()


if __name__ == "__main__":
    main()
