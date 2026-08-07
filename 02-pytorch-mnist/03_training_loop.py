"""3단계: MNIST 분류 모델의 학습/평가 루프를 작성한다.

이전 실습과의 연결
------------------
01_tensors_and_autograd.py 에서 autograd와 nn.Linear를 보았다.
02_mnist_dataset.py 에서 DataLoader로 이미지 배치를 꺼냈다.
이번 실습에서는 둘을 이어 01번 폴더 09번의 학습 루프를
PyTorch로 옮긴다.

    NumPy 09번                     PyTorch 이번 실습
    ---------------------------   -----------------------------
    forward(...)                  logits = model(x)
    mse_loss(...)                 loss = criterion(logits, labels)
    backward(...)                 loss.backward()
    update_parameters(...)        optimizer.step()

모델
----
이미지를 784칸 벡터로 펼친 뒤, 작은 MLP로 숫자 0~9를 분류한다.

    x: (batch, 784)
      -> Linear(784, 128) + ReLU
      -> Linear(128, 10)
      -> logits: (batch, 10)

출력 10개는 각 숫자 클래스의 점수(logit)다.
CrossEntropyLoss는 Softmax + 교차 엔트로피를 한 번에 계산한다.
(01번 폴더 06에서 본 분류 손실과 같은 역할)

학습과 평가
-----------
train 모드: 기울기를 계산하고 매개변수를 갱신한다.
eval 모드:  시험 데이터로 정확도만 측정하고 갱신하지 않는다.

이번 실습의 다음
----------------
학습 정확도와 시험 정확도의 차이를 더 자세히 보며
과적합(overfitting)과 대응을 살펴본다.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


DATA_DIR = Path(__file__).resolve().parent / "data"


def create_dataloaders(
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """MNIST 학습/시험 DataLoader를 만든다."""
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def create_model() -> nn.Module:
    """784 입력을 받아 10개 클래스 점수를 내는 MLP를 만든다."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def explain_mapping() -> None:
    """NumPy 학습 루프와 PyTorch API의 대응을 설명한다."""
    print("1. 09번 학습 루프와의 대응")
    print("   forward          -> model(x)")
    print("   loss 계산        -> criterion(logits, labels)")
    print("   backward         -> loss.backward()")
    print("   w = w - lr * dw  -> optimizer.step()")
    print("   기울기 비우기    -> optimizer.zero_grad()")
    print()


def explain_model(model: nn.Module) -> None:
    """모델 구조와 출력 shape를 확인한다."""
    device = next(model.parameters()).device
    dummy = torch.zeros(4, 1, 28, 28, device=device)
    logits = model(dummy)

    print("2. 모델 구조")
    print("   Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 10)")
    print(f"   device: {device}")
    print(f"   더미 입력 shape: {tuple(dummy.shape)}")
    print(f"   로짓 출력 shape: {tuple(logits.shape)}  # (batch, 클래스 수)")
    print("   nn.Flatten()이 view(batch, -1)과 같은 펼치기를 대신한다.\n")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """학습 데이터 한 epoch를 돌며 평균 손실을 반환한다."""
    model.train()
    total_loss = 0.0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return total_loss / sample_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """시험 데이터에서 평균 손실과 정확도를 계산한다."""
    model.eval()
    total_loss = 0.0
    correct = 0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        sample_count += batch_size

    average_loss = total_loss / sample_count
    accuracy = correct / sample_count
    return average_loss, accuracy


def show_sample_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """시험 배치 일부의 예측을 보여 준다."""
    model.eval()
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = model(images)
        predictions = logits.argmax(dim=1)

    print("5. 시험 배치 일부 예측")
    print("   정답:   ", labels[:10].tolist())
    print("   예측:   ", predictions[:10].tolist())
    matches = (predictions[:10] == labels[:10]).sum().item()
    print(f"   앞 10개 중 맞춘 개수: {matches}/10\n")


def explain_next_step() -> None:
    """과적합 주제로 연결한다."""
    print("6. 다음 실습")
    print("   학습 손실/정확도와 시험 손실/정확도를 비교한다.")
    print("   학습 데이터만 잘 맞추고 시험 데이터가 약하면 과적합이다.")


def main() -> None:
    explain_mapping()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = create_dataloaders(batch_size=64)
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    explain_model(model)

    print(f"3. 학습 시작 (device={device})")
    print("   epoch | train loss | test loss | test acc")
    print("   ------+------------+-----------+---------")

    epoch_count = 3
    for epoch in range(1, epoch_count + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_accuracy = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"   {epoch:>5} | {train_loss:>10.4f} | "
            f"{test_loss:>9.4f} | {test_accuracy:>7.2%}"
        )

    print()
    print("4. 관찰 포인트")
    print("   epoch가 지날수록 train/test 손실이 줄고 test 정확도가 오르면")
    print("   09번 XOR처럼 '데이터에 맞는 함수'를 찾아가는 중이다.")
    print("   CrossEntropyLoss + argmax가 분류 예측의 기본 패턴이다.\n")

    show_sample_predictions(model, test_loader, device)
    explain_next_step()


if __name__ == "__main__":
    main()
