"""
Простая нейросеть для текста: классификатор тональности на PyTorch.

Запуск обучения:
    python text_main.py

Запуск предсказания (после обучения):
    python text_main.py --predict "отличный фильм, мне очень понравился"
"""

import argparse
import os
import re

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/text_model.pth"

# ----- Учебный датасет: пары (текст, класс). 1 — позитив, 0 — негатив -----
TRAIN_DATA = [
    ("отличный фильм мне очень понравился", 1),
    ("прекрасная игра актеров восторг", 1),
    ("лучшая покупка за последнее время", 1),
    ("всё супер спасибо огромное", 1),
    ("замечательный сервис рекомендую", 1),
    ("мне безумно понравилось приду ещё", 1),
    ("великолепно просто нет слов", 1),
    ("хорошая книга увлекательный сюжет", 1),
    ("спасибо вам за помощь всё работает", 1),
    ("доволен покупкой отличное качество", 1),
    ("ужасный фильм потратил время зря", 0),
    ("полная ерунда разочарован", 0),
    ("плохой сервис больше не приду", 0),
    ("ненавижу эту отстойную штуку", 0),
    ("качество ужасное деньги на ветер", 0),
    ("скучно предсказуемо не советую", 0),
    ("отвратительный звук смотреть невозможно", 0),
    ("жалею о покупке верните деньги", 0),
    ("бездарный сюжет пустая трата времени", 0),
    ("всё сломалось через день ужас", 0),
]

# Дополнительно: без разметки (для теста)
TEST_PHRASES = [
    "мне понравился этот фильм",
    "что за ужасное приложение",
    "нормально но могло быть лучше",
]


def tokenize(text: str) -> list[str]:
    """Простой токенизатор: нижний регистр, только слова."""
    return re.findall(r"[а-яёa-z]+", text.lower())


def build_vocab(texts: list[str]) -> dict[str, int]:
    """Словарь: слово -> индекс. 0 — паддинг, 1 — <unk>."""
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int = 12) -> torch.Tensor:
    """Текст -> последовательность индексов фиксированной длины."""
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokenize(text)][:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class TextClassifier(nn.Module):
    """Embedding -> усреднение -> полносвязный слой."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)                    # [batch, seq_len, embed_dim]
        mask = (x != 0).unsqueeze(-1).float()      # не учитываем паддинг в среднем
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.net(pooled)                    # [batch, num_classes]


def train() -> None:
    vocab = build_vocab([t for t, _ in TRAIN_DATA])
    model = TextClassifier(len(vocab)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    xs = torch.stack([encode(t, vocab) for t, _ in TRAIN_DATA]).to(DEVICE)
    ys = torch.tensor([y for _, y in TRAIN_DATA], dtype=torch.long).to(DEVICE)

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(xs), ys)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            acc = (model(xs).argmax(dim=1) == ys).float().mean().item()
            print(f"Эпоха {epoch} — loss: {loss.item():.4f}, точность: {acc:.2%}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": vocab}, MODEL_PATH)
    print(f"Модель сохранена: {MODEL_PATH}")


def predict(texts: list[str]) -> None:
    saved = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    vocab = saved["vocab"]
    model = TextClassifier(len(vocab)).to(DEVICE)
    model.load_state_dict(saved["model"])
    model.eval()

    xs = torch.stack([encode(t, vocab) for t in texts]).to(DEVICE)
    with torch.no_grad():
        preds = model(xs).argmax(dim=1)
    for text, pred in zip(texts, preds):
        label = "позитив 🙂" if pred.item() == 1 else "негатив 🙁"
        print(f"«{text}» -> {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Классификатор тональности")
    parser.add_argument("--predict", nargs="+", help="Фразы для предсказания")
    args = parser.parse_args()

    if args.predict:
        predict(args.predict)
    else:
        train()
