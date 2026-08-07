"""2단계: MNIST 데이터셋과 DataLoader를 이해한다.

이전 실습과의 연결
------------------
01번 PyTorch 실습에서는 텐서, autograd, nn.Linear가
NumPy로 직접 구현했던 개념과 어떻게 대응하는지 보았다.
그때 입력은 우리가 손으로 만든 작은 텐서였다.

실제 학습에서는 데이터가 훨씬 많다. 이번 실습에서는
손글씨 숫자 데이터셋 MNIST를 내려받고, Dataset과 DataLoader가
배치를 어떻게 만들어 주는지 확인한다.

MNIST
-----
MNIST는 0~9 손글씨 숫자 이미지 데이터셋이다.

    학습용: 60,000장
    시험용: 10,000장
    이미지: 28 x 28 흑백
    정답: 0~9 중 하나의 클래스

01번 폴더 06에서 배운 classification(분류) 문제이며,
출력 뉴런 10개가 각 숫자 클래스의 점수가 된다.

Dataset과 DataLoader
--------------------
Dataset: 샘플 하나씩 꺼낼 수 있는 데이터 저장소에 가깝다.
         dataset[i] -> (이미지, 정답 라벨)

DataLoader: Dataset을 배치 단위로 묶어 반복하게 해 준다.
            01번 폴더 03_batch.py에서 여러 샘플을 한 행렬로
            묶었던 일을 자동화한다고 보면 된다.

    for images, labels in dataloader:
        # images.shape == (batch_size, 1, 28, 28)
        # labels.shape == (batch_size,)

이미지 텐서 shape의 1은 채널 수다. 흑백이라 채널이 1개다.
컬러 이미지(RGB)라면 보통 채널이 3이다.

이번 실습의 다음
----------------
데이터를 배치로 꺼낼 수 있게 되면, 다음 실습에서
모델 정의 + 학습/평가 루프를 연결한다.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_mnist_datasets() -> tuple[datasets.MNIST, datasets.MNIST]:
    """MNIST 학습/시험 분할을 내려받거나 캐시에서 불러온다."""
    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_dataset, test_dataset


def explain_mnist_overview(
    train_dataset: datasets.MNIST,
    test_dataset: datasets.MNIST,
) -> None:
    """데이터셋 크기와 한 샘플의 형태를 설명한다."""
    image, label = train_dataset[0]

    print("1. MNIST가 무엇인지")
    print("   손글씨 숫자 0~9를 분류하는 고전적인 이미지 데이터셋이다.")
    print(f"   학습 샘플 수: {len(train_dataset)}")
    print(f"   시험 샘플 수: {len(test_dataset)}")
    print(f"   저장 위치: {DATA_DIR}")
    print()

    print("2. 샘플 하나 살펴보기")
    print(f"   image 타입: {type(image)}")
    print(f"   image shape: {tuple(image.shape)}  # (channel, height, width)")
    print(f"   픽셀 값 범위: {image.min().item():.1f} ~ {image.max().item():.1f}")
    print("   ToTensor()가 0~255 정수를 0~1 실수로 바꾼다.")
    print(f"   label(정답 클래스): {label}")
    print("   label은 '이 이미지가 어떤 숫자인가'를 나타내는 클래스 번호다.\n")


def explain_dataset_vs_dataloader(
    train_dataset: datasets.MNIST,
) -> DataLoader:
    """Dataset과 DataLoader의 역할 차이를 배치 하나로 확인한다."""
    batch_size = 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    images, labels = next(iter(train_loader))

    print("3. Dataset vs DataLoader")
    print("   Dataset[i]  -> 샘플 1개")
    print("   DataLoader  -> 샘플을 batch_size개씩 묶어 줌")
    print(f"   batch_size = {batch_size}")
    print(f"   한 배치 이미지 shape: {tuple(images.shape)}")
    print("                     # (batch, channel, height, width)")
    print(f"   한 배치 라벨 shape:   {tuple(labels.shape)}")
    print(f"   이 배치의 라벨 예시:  {labels[:10].tolist()}")
    print("   shuffle=True면 매 epoch마다 샘플 순서를 섞어 암기를 줄인다.\n")
    return train_loader


def show_batch_as_model_input(images: torch.Tensor) -> None:
    """완전연결층에 넣기 위해 이미지를 펼치는 이유를 설명한다."""
    batch_size = images.shape[0]
    flat = images.view(batch_size, -1)

    print("4. 나중에 모델에 넣을 때의 shape")
    print("   지금 이미지는 2차원(28x28)이다.")
    print("   01번에서 쓴 Linear/Dense는 보통 1차원 벡터를 받는다.")
    print("   그래서 학습 루프에서는 이렇게 펼친다:")
    print("       x = images.view(batch_size, -1)")
    print(f"   펼친 결과 shape: {tuple(flat.shape)}  # (batch, 28*28=784)")
    print("   각 행이 '숫자 이미지 하나'를 784개 픽셀 특성으로 본 것이다.\n")


def count_batches(train_loader: DataLoader) -> None:
    """전체 학습 데이터를 배치로 나누면 몇 번 도는지 확인한다."""
    batch_count = len(train_loader)
    batch_size = train_loader.batch_size
    sample_count = len(train_loader.dataset)

    print("5. 한 epoch의 의미")
    print(f"   학습 샘플 {sample_count}개를 batch_size={batch_size}로 나누면")
    print(f"   약 {batch_count}개의 배치가 나온다.")
    print("   이 배치들을 한 바퀴 도는 것이 1 epoch이다.")
    print("   09번 XOR에서는 샘플이 4개뿐이라 배치=전체 데이터였다.")
    print("   MNIST는 데이터가 커서 배치로 나눠 여러 번 갱신한다.\n")


def explain_next_step() -> None:
    """다음 실습에서 학습 루프로 연결할 것을 안내한다."""
    print("6. 다음 실습")
    print("   DataLoader로 (images, labels) 배치를 반복하면서")
    print("   모델 순전파 -> 손실 -> backward -> optimizer 갱신을 수행한다.")
    print("   그게 PyTorch 버전의 09번 학습 루프다.")


def main() -> None:
    train_dataset, test_dataset = load_mnist_datasets()
    explain_mnist_overview(train_dataset, test_dataset)
    train_loader = explain_dataset_vs_dataloader(train_dataset)
    images, _labels = next(iter(train_loader))
    show_batch_as_model_input(images)
    count_batches(train_loader)
    explain_next_step()


if __name__ == "__main__":
    main()
