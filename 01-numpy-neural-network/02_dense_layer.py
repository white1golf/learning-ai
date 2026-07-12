"""2단계: 여러 뉴런을 묶어 완전연결층(Dense Layer)을 만든다.

모든 뉴런이 모든 입력을 받기 때문에 '완전연결층'이라고 부른다.
뉴런마다 자신만의 가중치 묶음과 편향 하나를 가진다.

입력 3개와 뉴런 2개를 사용하면:

    inputs.shape  == (3,)
    weights.shape == (2, 3)  # 뉴런 2개, 뉴런당 가중치 3개
    biases.shape  == (2,)    # 뉴런당 편향 1개
    outputs.shape == (2,)    # 뉴런마다 출력 1개
"""

import numpy as np


def calculate_each_neuron() -> np.ndarray:
    """뉴런을 하나씩 계산해 완전연결층 내부 연산을 확인한다."""
    inputs = np.array([1.0, 2.0, 3.0])
    weights = np.array(
        [
            [0.1, 0.2, 0.3],  # 뉴런 1의 가중치
            [0.4, 0.5, 0.6],  # 뉴런 2의 가중치
        ]
    )
    biases = np.array([0.5, -0.5])

    outputs = []

    print("1. 뉴런을 하나씩 계산")
    for neuron_number, (neuron_weights, bias) in enumerate(
        zip(weights, biases),
        start=1,
    ):
        weighted_sum = 0.0
        terms = []

        for input_value, weight in zip(inputs, neuron_weights):
            weighted_sum += input_value * weight
            terms.append(f"{input_value:.1f}×{weight:.1f}")

        output = weighted_sum + bias
        outputs.append(output)

        expression = " + ".join(terms)
        print(
            f"   뉴런 {neuron_number}: {expression} + 편향 {bias:.1f} "
            f"= {output:.1f}"
        )

    result = np.array(outputs)
    print(f"   층의 전체 출력: {result}\n")
    return result


def calculate_with_matrix() -> np.ndarray:
    """동일한 계산을 NumPy의 행렬-벡터 곱으로 한 번에 처리한다."""
    inputs = np.array([1.0, 2.0, 3.0])
    weights = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    biases = np.array([0.5, -0.5])

    outputs = weights @ inputs + biases

    print("2. 행렬 연산으로 한 번에 계산")
    print("   계산식: weights @ inputs + biases")
    print(f"   행렬 연산 결과: {outputs}\n")
    return outputs


def inspect_shapes() -> None:
    """배열 shape와 신경망 구조의 관계를 확인한다."""
    inputs = np.array([1.0, 2.0, 3.0])
    weights = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )
    biases = np.array([0.5, -0.5])
    outputs = weights @ inputs + biases

    print("3. shape 확인")
    print(f"   inputs.shape  = {inputs.shape}  -> 입력 특성 3개")
    print(f"   weights.shape = {weights.shape} -> 뉴런 2개 × 입력 3개")
    print(f"   biases.shape  = {biases.shape}  -> 뉴런별 편향 2개")
    print(f"   outputs.shape = {outputs.shape}  -> 뉴런별 출력 2개\n")


def main() -> None:
    outputs_by_loop = calculate_each_neuron()
    outputs_by_matrix = calculate_with_matrix()
    inspect_shapes()

    print("4. 두 계산 방법 검증")
    print(f"   결과가 같은가? {np.allclose(outputs_by_loop, outputs_by_matrix)}")


if __name__ == "__main__":
    main()
