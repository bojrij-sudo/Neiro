"""
Простая нейросеть на PyTorch: многослойный перцептрон для MNIST.

Запуск обучения:
    python main.py

Запуск предсказания (после обучения):
    python main.py --predict
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----- Конфигурация -----
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/model.pth"
DATA_DIR = "data"


# ----- Модель -----
class SimpleNet(nn.Module):
    """Многослойный перцептрон: 28x28 пикселей -> 10 классов (цифры 0-9)."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)


# ----- Данные -----
def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # среднее и std MNIST
        ]
    )
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    return train_loader, test_loader


# ----- Обучение -----
def train() -> None:
    train_loader, test_loader = get_dataloaders()
    model = SimpleNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # Точность на тестовой выборке
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Эпоха {epoch}/{EPOCHS} — точность на тесте: {accuracy:.2%}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")


# ----- Предсказание -----
def predict() -> None:
    model = SimpleNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    _, test_loader = get_dataloaders()
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        preds = model(images.to(DEVICE)).argmax(dim=1)

    for i in range(5):
        print(f"Предсказано: {preds[i]}, на самом деле: {labels[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Простая нейросеть для MNIST")
    parser.add_argument("--predict", action="store_true", help="Загрузить модель и сделать предсказания")
    args = parser.parse_args()

    if args.predict:
        predict()
    else:
        train()
