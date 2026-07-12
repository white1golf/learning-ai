"""1단계: 단일 뉴런의 가중치(weight)와 편향(bias)을 이해한다.

뉴런 하나의 기본 계산식:

    output = (input_1 * weight_1)
           + (input_2 * weight_2)
           + (input_3 * weight_3)
           + bias

가중치는 각 입력이 출력에 주는 영향의 크기와 방향을 나타낸다.
편향은 입력과 무관하게 최종 결과의 기준점을 이동시킨다.
"""

import numpy as np


def calculate_by_hand() -> None:
    """계산 과정을 항목별로 출력해 손계산과 연결한다."""
    inputs = [1.0, 2.0, 3.0]
    weights = [0.1, 0.2, 0.3]
    bias = 0.5

    weighted_values = [
        input_value * weight
        for input_value, weight in zip(inputs, weights)
    ]
    output = sum(weighted_values) + bias

    print("1. 손으로 풀어보는 계산")
    for index, (input_value, weight, value) in enumerate(
        zip(inputs, weights, weighted_values),
        start=1,
    ):
        print(f"   입력 {index}: {input_value} × 가중치 {weight} = {value:.1f}")
    print(f"   가중합: {sum(weighted_values):.1f}")
    print(f"   가중합 + 편향 {bias} = {output:.1f}\n")

def compare_loop_and_numpy() -> None:
    """반복문 계산과 NumPy 내적이 같은 연산임을 확인한다."""
    inputs = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.1, 0.2, 0.3])
    bias = 0.5

    output_with_loop = bias
    for input_value, weight in zip(inputs, weights):
        output_with_loop += input_value * weight

    output_with_numpy = np.dot(inputs, weights) + bias

    print("2. 반복문과 NumPy 비교")
    print(f"   반복문 결과: {output_with_loop}")
    print(f"   NumPy 결과: {output_with_numpy}")
    print(f"   두 결과가 같은가? {np.isclose(output_with_loop, output_with_numpy)}\n")


def experiment_with_parameters() -> None:
    """가중치와 편향을 바꿨을 때 출력이 어떻게 달라지는지 관찰한다."""
    inputs = np.array([1.0, 2.0, 3.0])

    experiments = [
        ("기본값", np.array([0.1, 0.2, 0.3]), 0.5),
        ("세 번째 입력 무시", np.array([0.1, 0.2, 0.0]), 0.5),
        ("세 번째 입력의 영향 반전", np.array([0.1, 0.2, -0.3]), 0.5),
        ("편향 제거", np.array([0.1, 0.2, 0.3]), 0.0),
    ]

    print("3. 가중치와 편향 변경 실험")
    for name, weights, bias in experiments:
        output = np.dot(inputs, weights) + bias
        print(
            f"   {name:<22} weights={weights}, "
            f"bias={bias:>4.1f} -> output={output:>4.1f}"
        )


def main() -> None:
    calculate_by_hand()
    compare_loop_and_numpy()
    experiment_with_parameters()


if __name__ == "__main__":
    main()
