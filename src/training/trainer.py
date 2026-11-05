"""Vòng lặp huấn luyện: Forward -> Loss -> Backward -> Optimize."""
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import evaluate


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Huấn luyện một epoch, trả về train loss và train accuracy trung bình."""
    device = device or next(model.parameters()).device
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()              # xóa gradient cũ
        logits = model(images)             # Forward
        loss = loss_fn(logits, labels)     # Loss
        loss.backward()                    # Backward
        optimizer.step()                   # Optimize

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("train_loader không có mẫu nào.")
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: Optional[DataLoader],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: Optional[torch.device] = None,
    writer=None,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    """Huấn luyện nhiều epoch; trả về history (list dict theo epoch).

    Mỗi phần tử: epoch, train_loss, train_accuracy, và test_loss/test_accuracy
    nếu có test_loader. `writer` là SummaryWriter (tùy chọn) để log TensorBoard.
    """
    device = device or next(model.parameters()).device
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_accuracy": tr["accuracy"]}
        if test_loader is not None:
            te = evaluate(model, test_loader, loss_fn, device)
            row.update(test_loss=te["loss"], test_accuracy=te["accuracy"])
        history.append(row)

        if writer is not None:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
        if verbose:
            print(" | ".join(f"{k}={v:.4f}" if k != "epoch" else f"epoch {v}" for k, v in row.items()))

    return history
