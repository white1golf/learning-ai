"""6단계: 예측과 정답의 차이를 손실 함수로 측정한다.

이전 실습과의 연결
------------------
05_two_layer_forward.py에서는 weights(가중치)와 biases(편향)를 사람이
임의로 정한 뒤, 입력이 두 층을 통과해 logits(클래스 점수)가 되는
forward pass(순전파)를 계산했다. 이 값들은 아직 학습된 값이 아니다.

weights와 biases는 신경망이 학습해야 하는 parameters(매개변수)다.
학습하려면 먼저 현재 매개변수가 만든 예측이 얼마나 좋은지 측정할 기준이
필요하다. 손실 함수(loss function)가 예측과 정답의 차이를 하나의 숫자로
만들어 그 기준을 제공한다.

    손실이 작다 -> 예측이 정답에 가깝다
    손실이 크다 -> 예측이 정답에서 멀다

전체 학습 과정에서 이번 실습의 위치는 다음과 같다.

    1. 매개변수에 초기값을 넣는다.             <- 05번에서 임의로 설정
    2. 순전파로 예측값을 계산한다.             <- 05번에서 계산
    3. 예측값과 정답으로 손실을 계산한다.      <- 이번 06번
    4. 손실을 줄일 매개변수의 변화량을 구한다. <- 다음 실습부터
    5. 매개변수를 갱신하고 2~4를 반복한다.     <- 이 반복이 학습

이번 예제에는 실제 데이터셋이 없으므로 입력의 정답 클래스도 학습 흐름을
관찰하기 위해 임의로 정한다. 즉, 가중치와 정답 모두 교육용 예제 값이며,
목적은 현재 매개변수의 품질을 손실로 평가하는 과정을 이해하는 것이다.

회귀, 분류, 클래스
------------------
supervised learning(지도 학습)에서는 보통 입력과 정답이 한 쌍을 이룬다.
입력 하나를 sample(샘플), 그 샘플에 대응하는 정답을 target(타깃) 또는
label(레이블)이라고 부른다. 예측하려는 정답의 종류에 따라 대표적으로
regression(회귀)과 classification(분류)으로 문제를 나눌 수 있다.

regression(회귀)은 집값, 온도, 이동 시간처럼 연속적인 숫자를 예측한다.
예를 들어 집의 면적과 방 개수를 입력받아 4억 2천만 원이라는 값을
출력하는 문제다. 예측 숫자와 실제 숫자가 얼마나 떨어졌는지 측정할 수
있으며, 이번 실습에서는 mean squared error(MSE, 평균 제곱 오차)를 쓴다.

classification(분류)은 미리 정해 둔 여러 범주 중 하나를 고른다.
이때 각각의 범주를 class(클래스)라고 부른다. 예를 들어 사진을 고양이와
강아지 중 하나로 분류한다면 '고양이'와 '강아지'가 각각 클래스다.
숫자 0과 1은 클래스 이름 대신 사용하는 식별 번호일 뿐이다. 클래스 1이
클래스 0보다 크거나 좋다는 뜻은 아니다.

05번에는 출력 뉴런이 2개 있었지만 그 출력의 의미를 정하지 않았다.
이번 분류 예제에서는 학습 흐름을 관찰하기 위해 다음과 같이 가정한다.

    출력 뉴런 0 -> 클래스 0의 점수
    출력 뉴런 1 -> 클래스 1의 점수

각 출력은 어느 클래스가 더 어울리는지 나타내는 logit(로짓), 즉 가공되지
않은 점수다. Softmax(소프트맥스)는 이 점수들을 확률로 바꾸고,
cross-entropy(교차 엔트로피)는 정답 클래스에 부여한 확률을 손실로 바꾼다.

이번 실습에서는 서로 다른 두 문제의 손실을 별도 예제로 다룬다.

1. 회귀 예제: 평균 제곱 오차(MSE)
   연속적인 숫자인 예측값과 정답의 차이를 측정한다.

       MSE = mean((prediction - target) ** 2)

2. 분류 예제: Softmax와 범주형 교차 엔트로피(Cross-Entropy)
   05번의 출력 2개를 클래스 2개의 점수라고 가정해 손실을 측정한다.

       probability = softmax(logits)
       loss = -log(정답 클래스의 확률)

손실 값만으로는 아직 어떤 가중치를 어떻게 바꿔야 하는지 알 수 없다.
다음 실습에서는 수치 미분으로 매개변수를 조금 바꿨을 때 손실이 어떻게
변하는지 확인하고, 손실이 작아지는 방향을 찾는다.
"""

import numpy as np


def mean_squared_error(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """예측과 정답 차이의 제곱을 평균낸다."""
    squared_errors = (predictions - targets) ** 2
    return float(np.mean(squared_errors))


def softmax(logits: np.ndarray) -> np.ndarray:
    """클래스 점수를 각 행의 합이 1인 확률로 변환한다.

    지수 함수가 너무 큰 값을 만들지 않도록 각 행의 최댓값을 먼저 뺀다.
    모든 값에서 같은 수를 빼도 Softmax 결과는 변하지 않는다.
    """
    stabilized_logits = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(stabilized_logits)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def categorical_cross_entropy(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, float]:
    """샘플별 교차 엔트로피와 배치의 평균 손실을 반환한다.

    targets에는 one-hot 배열 대신 정답 클래스의 번호를 저장한다.
    예를 들어 targets가 [1, 0]이면 첫 샘플의 정답은 클래스 1이고,
    두 번째 샘플의 정답은 클래스 0이다.
    """
    sample_indices = np.arange(len(probabilities))
    correct_class_probabilities = probabilities[sample_indices, targets]

    # log(0)은 정의되지 않으므로 확률이 정확히 0이 되지 않게 보호한다.
    safe_probabilities = np.clip(
        correct_class_probabilities,
        1e-7,
        1.0,
    )
    sample_losses = -np.log(safe_probabilities)
    mean_loss = float(np.mean(sample_losses))
    return sample_losses, mean_loss


def explain_learning_context() -> None:
    """05번의 순전파가 손실 계산과 학습으로 이어지는 맥락을 설명한다."""
    print("1. 지금까지의 계산과 앞으로의 학습")
    print("   05번의 weights와 biases는 학습 결과가 아니라 임의의 초기값이다.")
    print("   그 값으로 순전파를 수행해 예측값인 logits를 얻었다.")
    print("   이번에는 logits와 정답을 비교해 현재 매개변수를 평가한다.")
    print("   다음부터는 손실이 작아지는 방향으로 매개변수를 수정한다.")
    print("   순전파 -> 손실 계산 -> 기울기 계산 -> 매개변수 갱신 -> 반복\n")


def explain_problem_types() -> None:
    """회귀, 분류, 클래스와 두 손실 예제의 관계를 설명한다."""
    print("2. 회귀 문제와 분류 문제")
    print("   지도 학습에서는 입력 샘플마다 모델이 맞혀야 할 정답이 있다.")
    print("   회귀: 집값이나 온도처럼 연속적인 숫자를 예측한다.")
    print("   분류: 미리 정한 여러 범주 중 하나를 선택한다.")
    print("   클래스: 분류 문제에서 선택할 수 있는 각각의 범주다.")
    print("   예: 고양이=클래스 0, 강아지=클래스 1")
    print("   여기서 0과 1은 범주의 식별 번호이며 크기나 순서를 뜻하지 않는다.")
    print("   다음 MSE는 회귀 예제이고, 이후 Softmax는 별도의 분류 예제다.\n")


def explain_mean_squared_error() -> None:
    """MSE를 오차, 제곱, 평균의 순서로 나누어 확인한다."""
    predictions = np.array([2.5, 0.0, 2.1, 7.8])
    targets = np.array([3.0, -0.5, 2.0, 7.0])

    errors = predictions - targets
    squared_errors = errors**2
    loss = mean_squared_error(predictions, targets)

    print("3. 회귀 예제: 평균 제곱 오차(MSE)")
    print("   네 샘플에서 하나씩 연속적인 숫자를 예측했다고 가정한다.")
    print(f"   예측:          {predictions}")
    print(f"   정답:          {targets}")
    print(f"   예측 - 정답:   {errors}")
    print(f"   오차의 제곱:   {squared_errors}")
    print(f"   제곱 오차 평균: {loss:.4f}")
    print("   큰 오차일수록 제곱 때문에 손실에 더 크게 반영된다.\n")


def explain_first_softmax() -> None:
    """첫 번째 샘플의 logits가 확률로 변환되는 계산을 풀어본다."""
    logits = np.array([[0.31, 0.71]])
    target = 1

    stabilized = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(stabilized)
    exponential_sum = np.sum(exponentials, axis=1, keepdims=True)
    probabilities = exponentials / exponential_sum
    correct_probability = probabilities[0, target]
    loss = -np.log(correct_probability)

    print("4. 분류 예제: 첫 번째 샘플의 Softmax와 교차 엔트로피")
    print("   05번의 출력 뉴런 0과 1을 클래스 0과 1의 점수로 가정한다.")
    print(f"   원래 클래스 점수 logits: {logits[0]}")
    print(f"   최댓값을 뺀 점수:        {stabilized[0]}")
    print(f"   지수 함수 적용:           {np.round(exponentials[0], 4)}")
    print(f"   지수 값의 합:             {exponential_sum[0, 0]:.4f}")
    print(f"   Softmax 확률:             {np.round(probabilities[0], 4)}")
    print(f"   확률의 합:                {np.sum(probabilities[0]):.1f}")
    print(f"   정답 클래스:              {target}")
    print(f"   정답 클래스의 확률:       {correct_probability:.4f}")
    print(f"   -log(정답 확률):          {loss:.4f}\n")


def evaluate_two_layer_outputs() -> None:
    """5단계 신경망의 출력과 임의의 정답으로 분류 손실을 계산한다."""
    # 05번에서 임의의 매개변수로 계산한 두 출력 뉴런의 점수다.
    # 같은 순전파를 반복하지 않고, 두 실습의 연결을 보기 위해 결과를 옮겼다.
    logits = np.array(
        [
            [0.31, 0.71],
            [0.57, 1.17],
            [0.48, -0.12],
            [0.47, -0.37],
        ]
    )

    # 실제 학습에서는 데이터셋이 정답을 제공한다. 지금은 입력 샘플마다
    # 어떤 클래스가 정답이라고 가정할지 교육용으로 직접 지정한다.
    # 숫자는 클래스의 식별 번호일 뿐 크기나 순위를 의미하지 않는다.
    targets = np.array([1, 1, 0, 0])

    probabilities = softmax(logits)
    sample_losses, mean_loss = categorical_cross_entropy(
        probabilities,
        targets,
    )
    predictions = np.argmax(probabilities, axis=1)
    accuracy = np.mean(predictions == targets)

    print("5. 05번 순전파 결과를 분류 손실로 연결")
    print("   아래 logits는 05번의 임의 매개변수가 만든 예측값이다.")
    print("   targets는 이번 실습을 위해 가정한 정답 클래스다.")
    print(f"   logits:\n{logits}")
    print(f"\n   Softmax 확률:\n{np.round(probabilities, 4)}")
    print(f"\n   정답 클래스:       {targets}")
    print(f"   예측 클래스:       {predictions}")
    print(f"   샘플별 손실:       {np.round(sample_losses, 4)}")
    print(f"   배치 평균 손실:    {mean_loss:.4f}")
    print(f"   정확도:            {accuracy:.2%}\n")

    print("6. 정확도와 손실의 차이")
    print("   모든 클래스를 맞혀도 정답 확률이 1이 아니면 손실은 0보다 크다.")
    print("   손실은 정답 여부뿐 아니라 예측의 확신 정도까지 표현한다.")
    print("   다음 목표는 이 평균 손실이 줄어들도록 매개변수를 바꾸는 것이다.")


def main() -> None:
    explain_learning_context()
    explain_problem_types()
    explain_mean_squared_error()
    explain_first_softmax()
    evaluate_two_layer_outputs()


if __name__ == "__main__":
    main()
