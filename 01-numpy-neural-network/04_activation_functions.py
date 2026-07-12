"""4단계: 활성화 함수로 신경망에 비선형성을 추가한다.

뉴런의 계산은 두 부분으로 나눌 수 있다.

    z = inputs @ weights.T + biases  # 선형 계산
    output = activation(z)           # 활성화 함수

선형 계산만 여러 층 쌓으면 결국 하나의 선형 계산으로 합칠 수 있다.
활성화 함수는 신경망이 곡선이나 복잡한 경계를 표현하도록 만든다.
"""

import numpy as np


def step(values: np.ndarray) -> np.ndarray:
    """0보다 크면 1, 그렇지 않으면 0을 반환한다.

    초기 퍼셉트론에서 사용했지만 기울기가 대부분 0이어서
    현대 신경망의 경사하강법에는 적합하지 않다.
    """
    return np.where(values > 0, 1.0, 0.0)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """모든 값을 0과 1 사이의 부드러운 값으로 변환한다."""
    return 1.0 / (1.0 + np.exp(-values))


def relu(values: np.ndarray) -> np.ndarray:
    """음수는 0으로 만들고 양수는 그대로 통과시킨다."""
    return np.maximum(0.0, values)


def compare_activation_functions() -> None:
    """같은 값에 세 활성화 함수를 적용해 차이를 비교한다."""
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    print("1. 같은 입력에 서로 다른 활성화 함수 적용")
    print(f"   입력:    {values}")
    print(f"   Step:    {step(values)}")
    print(f"   Sigmoid: {np.round(sigmoid(values), 4)}")
    print(f"   ReLU:    {relu(values)}\n")


def activate_dense_layer() -> None:
    """이전 실습의 배치 출력을 ReLU에 통과시킨다."""
    inputs = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [0.5, 1.0, 1.5],
            [-1.0, 0.0, 1.0],
        ]
    )
    weights = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [-0.3, 0.2, -0.1],
        ]
    )
    biases = np.array([0.5, -0.5, 0.1])

    linear_outputs = inputs @ weights.T + biases
    activated_outputs = relu(linear_outputs)

    print("2. 완전연결층 뒤에 ReLU 연결")
    print(f"   선형 출력 z:\n{linear_outputs}")
    print(f"\n   ReLU 적용 결과:\n{activated_outputs}\n")


def show_why_nonlinearity_matters() -> None:
    """선형 함수 두 개를 합쳐도 다시 선형 함수가 됨을 확인한다."""
    inputs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # 첫 번째 선형층: y = 2x + 1
    first_layer = 2.0 * inputs + 1.0
    # 두 번째 선형층: y = 3x - 2
    two_linear_layers = 3.0 * first_layer - 2.0
    # 두 식을 합치면 y = 3(2x + 1) - 2 = 6x + 1
    combined_layer = 6.0 * inputs + 1.0

    with_relu = 3.0 * relu(first_layer) - 2.0

    print("3. 비선형성이 필요한 이유")
    print(f"   입력:                 {inputs}")
    print(f"   선형층 두 개:          {two_linear_layers}")
    print(f"   하나로 합친 선형층:     {combined_layer}")
    print(
        "   두 선형 결과가 같은가? "
        f"{np.allclose(two_linear_layers, combined_layer)}"
    )
    print(f"   중간에 ReLU를 넣은 결과: {with_relu}")
    print("   ReLU가 들어간 결과는 하나의 직선으로 합칠 수 없다.")


def main() -> None:
    compare_activation_functions()
    activate_dense_layer()
    show_why_nonlinearity_matters()


if __name__ == "__main__":
    main()
