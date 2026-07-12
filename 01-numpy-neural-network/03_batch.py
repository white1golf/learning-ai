"""3단계: 여러 입력 샘플을 배치(batch)로 한 번에 처리한다.

이전 실습에서는 하나의 입력 샘플을 뉴런 2개가 처리했다.
이번에는 입력 샘플 4개를 동일한 뉴런 2개가 한꺼번에 처리한다.

    inputs.shape  == (4, 3)  # 샘플 4개, 샘플당 특성 3개
    weights.shape == (2, 3)  # 뉴런 2개, 뉴런당 가중치 3개
    biases.shape  == (2,)    # 뉴런당 편향 1개
    outputs.shape == (4, 2)  # 샘플 4개, 샘플당 출력 2개

배치는 모델이 여러 데이터를 효율적으로 학습하도록 묶은 데이터 단위다.
"""

import numpy as np


def calculate_sample_by_sample(
    inputs: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    """각 샘플을 반복문으로 처리해 배치 계산의 의미를 확인한다."""
    outputs = []

    print("1. 입력 샘플을 하나씩 처리")
    for sample_number, sample in enumerate(inputs, start=1):
        sample_output = weights @ sample + biases
        outputs.append(sample_output)
        print(f"   샘플 {sample_number}: {sample} -> {sample_output}")

    result = np.array(outputs)
    print(f"   전체 출력 shape: {result.shape}\n")
    return result


def calculate_as_batch(
    inputs: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    """모든 샘플을 하나의 행렬 곱으로 처리한다."""
    # inputs는 각 행이 샘플이고 weights는 각 행이 뉴런이다.
    # 행렬 곱의 안쪽 크기를 맞추기 위해 weights를 전치한다.
    # (4, 3) @ (3, 2) -> (4, 2)
    outputs = inputs @ weights.T + biases

    print("2. 배치 행렬 연산")
    print("   계산식: inputs @ weights.T + biases")
    print(f"   배치 출력:\n{outputs}\n")
    return outputs


def inspect_shapes(
    inputs: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    outputs: np.ndarray,
) -> None:
    """각 배열의 축이 신경망에서 무엇을 뜻하는지 확인한다."""
    print("3. shape와 축의 의미")
    print(f"   inputs.shape   = {inputs.shape}  -> (샘플 수, 입력 특성 수)")
    print(f"   weights.shape  = {weights.shape}  -> (뉴런 수, 입력 특성 수)")
    print(f"   weights.T.shape = {weights.T.shape}  -> 행렬 곱을 위해 전치")
    print(f"   biases.shape   = {biases.shape}  -> (뉴런 수,)")
    print(f"   outputs.shape  = {outputs.shape}  -> (샘플 수, 뉴런 수)\n")


def explain_bias_broadcasting(
    inputs: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
) -> None:
    """편향 벡터가 모든 샘플에 자동으로 더해지는 과정을 보여준다."""
    weighted_sums = inputs @ weights.T

    print("4. 편향 브로드캐스팅")
    print(f"   편향은 뉴런마다 하나: {biases}")
    print("   NumPy가 각 샘플의 가중합에 같은 편향을 더한다.")
    print(f"   편향을 더하기 전 첫 샘플: {weighted_sums[0]}")
    print(f"   편향을 더한 후 첫 샘플:  {weighted_sums[0] + biases}\n")


def main() -> None:
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
        ]
    )
    biases = np.array([0.5, -0.5])

    outputs_by_loop = calculate_sample_by_sample(inputs, weights, biases)
    outputs_by_batch = calculate_as_batch(inputs, weights, biases)
    inspect_shapes(inputs, weights, biases, outputs_by_batch)
    explain_bias_broadcasting(inputs, weights, biases)

    print("5. 두 계산 방법 비교")
    print(f"   결과가 같은가? {np.allclose(outputs_by_loop, outputs_by_batch)}")


if __name__ == "__main__":
    main()
