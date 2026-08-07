"""PyTorch 환경이 준비되었는지 확인하는 실행 파일."""

import platform

import torch


def main() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0]])
    w = torch.tensor(
        [
            [0.1, 0.4],
            [0.2, 0.5],
            [0.3, 0.6],
        ]
    )
    y = x @ w

    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"입력 shape: {tuple(x.shape)}")
    print(f"가중치 shape: {tuple(w.shape)}")
    print(f"출력: {y}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    print("환경 준비 완료!")


if __name__ == "__main__":
    main()
