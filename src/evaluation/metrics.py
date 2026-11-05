"""Đánh giá mô hình trên dữ liệu không dùng để cập nhật trọng số."""
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Tỉ lệ dự đoán đúng của một batch, trong khoảng [0, 1]."""
    if labels.numel() == 0:
        raise ValueError("Batch rỗng, không thể tính accuracy.")
    return (logits.argmax(dim=1) == labels).float().mean().item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Trả về {'loss': ..., 'accuracy': ...} trên toàn bộ data_loader.

    Gọi model.eval() và torch.no_grad(); trả model về chế độ trước đó khi xong.
    """
    device = device or next(model.parameters()).device
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()

    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    model.train(was_training)
    if total == 0:
        raise ValueError("data_loader không có mẫu nào.")
    return {"loss": total_loss / total, "accuracy": total_correct / total}


@torch.no_grad()
def predict_all(model, data_loader, device=None):
    """Trả về (nhãn thật, nhãn dự đoán) dạng tensor CPU, dùng cho confusion matrix."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    y_true, y_pred = [], []
    for images, labels in data_loader:
        y_pred.append(model(images.to(device)).argmax(dim=1).cpu())
        y_true.append(labels.cpu())
    model.train(was_training)
    return torch.cat(y_true), torch.cat(y_pred)
