"""5단계: 두 개의 완전연결층을 연결해 순전파를 수행한다.

순전파(forward pass)는 입력이 각 층을 차례로 지나 최종 출력이 되는 과정이다.
이번 신경망의 계산 흐름은 다음과 같다.

    입력 X
      -> 첫 번째 완전연결층: Z1 = X @ W1.T + b1
      -> ReLU 활성화:       A1 = ReLU(Z1)
      -> 두 번째 완전연결층: Z2 = A1 @ W2.T + b2
      -> 최종 출력 Z2

배치 크기가 4이고 입력 특성, 은닉 뉴런, 출력 뉴런 수가 각각 3, 3, 2이면:

    X.shape  == (4, 3)
    W1.shape == (3, 3)
    Z1.shape == (4, 3)
    A1.shape == (4, 3)
    W2.shape == (2, 3)
    Z2.shape == (4, 2)

두 번째 층에는 아직 활성화 함수를 적용하지 않는다. 최종 출력의 해석과
활성화 함수는 회귀인지 분류인지, 어떤 손실 함수를 쓰는지에 따라 달라진다.
"""

import numpy as np


def relu(values: np.ndarray) -> np.ndarray:
    """음수는 0으로 만들고 양수는 그대로 통과시킨다."""
    return np.maximum(0.0, values)


def dense_forward(
    inputs: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    """입력 배치가 완전연결층을 통과한 선형 출력을 반환한다."""
    return inputs @ weights.T + biases


def create_network_values() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """예제에서 사용할 입력 배치와 두 층의 매개변수를 만든다."""
    inputs = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [0.5, 1.0, 1.5],
            [-1.0, 0.0, 1.0],
        ]
    )

    first_weights = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [-0.3, 0.2, -0.1],
        ]
    )
    first_biases = np.array([0.5, -0.5, 0.0])

    second_weights = np.array(
        [
            [0.5, -0.2, 0.1],
            [-0.3, 0.4, 0.2],
        ]
    )
    second_biases = np.array([0.1, -0.2])

    return (
        inputs,
        first_weights,
        first_biases,
        second_weights,
        second_biases,
    )


def explain_first_sample() -> np.ndarray:
    """첫 번째 입력 샘플의 순전파를 뉴런별로 풀어서 계산한다."""
    (
        inputs,
        first_weights,
        first_biases,
        second_weights,
        second_biases,
    ) = create_network_values()
    sample = inputs[0]

    print("1. 첫 번째 샘플을 첫 번째 층에 전달")
    print(f"   입력: {sample}")

    first_linear_outputs = []
    for neuron_number, (weights, bias) in enumerate(
        zip(first_weights, first_biases),
        start=1,
    ):
        weighted_sum = float(sample @ weights)
        output = weighted_sum + bias
        first_linear_outputs.append(output)

        terms = [
            f"{input_value:.1f}×{weight:.1f}"
            for input_value, weight in zip(sample, weights)
        ]
        print(
            f"   은닉 뉴런 {neuron_number}: {' + '.join(terms)} "
            f"+ 편향 {bias:.1f} = {output:.2f}"
        )

    first_linear = np.array(first_linear_outputs)
    first_activated = relu(first_linear)
    print(f"   첫 번째 층의 선형 출력 Z1: {first_linear}")
    print(f"   ReLU를 적용한 은닉 출력 A1: {first_activated}\n")

    print("2. 은닉 출력을 두 번째 층에 전달")
    final_outputs = []
    for neuron_number, (weights, bias) in enumerate(
        zip(second_weights, second_biases),
        start=1,
    ):
        weighted_sum = float(first_activated @ weights)
        output = weighted_sum + bias
        final_outputs.append(output)

        terms = [
            f"{hidden_value:.2f}×{weight:.1f}"
            for hidden_value, weight in zip(first_activated, weights)
        ]
        print(
            f"   출력 뉴런 {neuron_number}: {' + '.join(terms)} "
            f"+ 편향 {bias:.1f} = {output:.2f}"
        )

    result = np.array(final_outputs)
    print(f"   첫 번째 샘플의 최종 출력 Z2: {result}\n")
    return result


def forward_batch() -> np.ndarray:
    """같은 두 층의 순전파를 전체 입력 배치에 행렬 연산으로 수행한다."""
    (
        inputs,
        first_weights,
        first_biases,
        second_weights,
        second_biases,
    ) = create_network_values()

    first_linear = dense_forward(inputs, first_weights, first_biases)
    first_activated = relu(first_linear)
    final_outputs = dense_forward(
        first_activated,
        second_weights,
        second_biases,
    )

    print("3. 전체 배치의 순전파")
    print(f"   첫 번째 층의 선형 출력 Z1:\n{first_linear}")
    print(f"\n   ReLU를 적용한 은닉 출력 A1:\n{first_activated}")
    print(f"\n   두 번째 층의 최종 출력 Z2:\n{final_outputs}\n")

    print("4. 각 단계의 shape")
    print(f"   입력 X:       {inputs.shape}")
    print(f"   가중치 W1:    {first_weights.shape}")
    print(f"   선형 출력 Z1: {first_linear.shape}")
    print(f"   은닉 출력 A1: {first_activated.shape}")
    print(f"   가중치 W2:    {second_weights.shape}")
    print(f"   최종 출력 Z2: {final_outputs.shape}\n")

    return final_outputs


def main() -> None:
    first_sample_by_steps = explain_first_sample()
    batch_outputs = forward_batch()

    print("5. 단계별 계산과 배치 계산 비교")
    print(f"   결과가 같은가? {np.allclose(first_sample_by_steps, batch_outputs[0])}")


if __name__ == "__main__":
    main()
