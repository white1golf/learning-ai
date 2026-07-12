"""첫 실습 환경이 준비되었는지 확인하는 실행 파일."""

import platform

import numpy as np


def main() -> None:
    inputs = np.array([[1.0, 2.0, 3.0]])
    weights = np.array(
        [
            [0.1, 0.4],
            [0.2, 0.5],
            [0.3, 0.6],
        ]
    )
    outputs = inputs @ weights

    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")
    print(f"입력 shape: {inputs.shape}")
    print(f"가중치 shape: {weights.shape}")
    print(f"출력: {outputs}")
    print("환경 준비 완료!")


if __name__ == "__main__":
    main()
