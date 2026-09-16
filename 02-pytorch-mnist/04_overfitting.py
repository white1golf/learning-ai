"""4단계: 과적합이 무엇인지 보고, Dropout 등으로 완화한다.

이전 실습과의 연결
------------------
03번에서는 학습 루프로 MNIST 분류기를 학습했다. 그때도 train loss와
test loss를 같이 출력했지만, MNIST는 비교적 쉬운 편이라 차이가
작아 보일 수 있다.

이번 실습에서는 상황을 조금 과장한다.

    1. 학습 데이터를 아주 조금만 쓴다. (예: 1000장)
    2. 매개변수는 상대적으로 많은 넓은 MLP를 쓴다.
    3. epoch를 충분히 돌린다.

그러면 모델이 학습 샘플을 거의 외워 버려, 학습 정확도는 높은데
시험 정확도는 상대적으로 낮아지는 현상을 관찰하기 쉽다.
이것이 overfitting(과적합)이다.

과적합
------
과적합: 학습 데이터에는 잘 맞지만, 아직 보지 못한 데이터에는
         일반화가 약한 상태.

대응 예시 (이번 실습에서 비교):

    - Dropout: 학습 중 뉴런 일부를 무작위로 꺼서 암기를 방해
    - weight decay: 가중치가 너무 커지지 않게 손실에 규제를 추가
                    (optimizer의 weight_decay 인자)

중요: Dropout은 model.train()일 때만 작동하고,
      model.eval()에서는 꺼진다. 그래서 평가/추론 모드가 필요하다.

이번 단계로 02번 폴더의 로드맵
(데이터 -> 학습 루프 -> 과적합)이 한 바퀴 돈다.
다음은 상위 README의 03. 언어 모델(Bigram) 단계다.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor


DATA_DIR = Path(__file__).resolve().parent / "data"


def create_dataloaders(
    train_sample_count: int = 1000,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """학습 데이터를 일부만 써서 과적합이 잘 보이게 한다."""
    full_train = datasets.MNIST(
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

    small_train = Subset(full_train, list(range(train_sample_count)))
    train_loader = DataLoader(small_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def create_wide_mlp(dropout_rate: float = 0.0) -> nn.Module:
    """매개변수가 많은 MLP. dropout_rate>0이면 Dropout을 넣는다."""
    layers: list[nn.Module] = [
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
    ]
    if dropout_rate > 0.0:
        layers.append(nn.Dropout(dropout_rate))
    layers.extend(
        [
            nn.Linear(512, 512),
            nn.ReLU(),
        ]
    )
    if dropout_rate > 0.0:
        layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(512, 10))
    return nn.Sequential(*layers)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """한 epoch 학습 후 평균 손실과 정확도를 반환한다."""
    model.train()
    total_loss = 0.0
    correct = 0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        sample_count += batch_size

    return total_loss / sample_count, correct / sample_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """평가 모드에서 손실과 정확도를 계산한다."""
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

    return total_loss / sample_count, correct / sample_count


def run_experiment(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    weight_decay: float,
    epoch_count: int,
) -> None:
    """한 설정을 학습하며 train/test 정확도 차이를 출력한다."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.05,
        weight_decay=weight_decay,
    )

    print(f"   [{name}]")
    print("   epoch | train acc | test acc | gap(train-test)")
    print("   ------+-----------+----------+----------------")

    for epoch in range(1, epoch_count + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        gap = train_acc - test_acc
        if epoch == 1 or epoch == epoch_count or epoch % 5 == 0:
            print(
                f"   {epoch:>5} | {train_acc:>8.2%} | "
                f"{test_acc:>7.2%} | {gap:>14.2%}"
            )

    print(
        f"   마지막 train loss={train_loss:.4f}, "
        f"test loss={test_loss:.4f}\n"
    )


def explain_overfitting() -> None:
    """과적합 개념을 먼저 설명한다."""
    print("1. 과적합이란")
    print("   학습 데이터에는 잘 맞지만, 새 데이터에는 약한 상태다.")
    print("   신호보다 학습셋의 세부/노이즈까지 외운 것에 가깝다.")
    print("   관찰 지표: train 정확도 >> test 정확도 (gap이 큼)\n")

    print("2. 이번 실험 설계")
    print("   학습 샘플을 1000장만 쓰고, 넓은 MLP로 충분히 학습한다.")
    print("   A) 규제 없음")
    print("   B) Dropout + weight decay")
    print("   B가 A보다 gap이 작거나 test 정확도가 더 나은지 비교한다.\n")


def explain_next_stage() -> None:
    """02번 폴더 종료와 다음 큰 단계를 안내한다."""
    print("4. 정리와 다음 단계")
    print("   02번에서 Dataset/DataLoader, 학습 루프, 과적합 대응을 봤다.")
    print("   다음 큰 단계는 03. 문자 단위 Bigram 언어 모델이다.")
    print("   이미지 분류 대신 '다음 글자 예측'으로 넘어간다.")


def main() -> None:
    explain_overfitting()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = create_dataloaders(train_sample_count=1000)
    epoch_count = 15

    print(f"3. 실험 (device={device}, train samples=1000, epochs={epoch_count})")
    torch.manual_seed(0)
    overfit_model = create_wide_mlp(dropout_rate=0.0).to(device)
    run_experiment(
        name="A. 규제 없음",
        model=overfit_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        weight_decay=0.0,
        epoch_count=epoch_count,
    )

    torch.manual_seed(0)
    regularized_model = create_wide_mlp(dropout_rate=0.5).to(device)
    run_experiment(
        name="B. Dropout(0.5) + weight_decay",
        model=regularized_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        weight_decay=1e-3,
        epoch_count=epoch_count,
    )

    print("   관찰 포인트")
    print("   - A는 train acc가 빨리 높아지고 gap이 커지기 쉽다.")
    print("   - B는 학습이 조금 느려도 test 쪽으로 더 안정적인 경우가 많다.")
    print("   - Dropout은 train()에서만 켜지고 eval()에서는 꺼진다.\n")
    explain_next_stage()


if __name__ == "__main__":
    main()
