from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    CLASS_NAMES,
    FIGURES_DIR,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    PREDICTIONS_DIR,
)


PathLike = Union[str, Path]


def _get_model_device(model: nn.Module, device: Optional[torch.device]) -> torch.device:
    """Trả về device truyền vào hoặc device của parameter đầu tiên của model."""
    if device is not None:
        return torch.device(device)

    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("Model không có parameter để xác định device.") from error


def _class_name(class_index: int, class_names: Sequence[str]) -> str:
    """Trả về tên lớp; vẫn hiển thị được nếu index nằm ngoài danh sách lớp."""
    if 0 <= class_index < len(class_names):
        return class_names[class_index]
    return f"Class {class_index}"


def _denormalize(images: torch.Tensor) -> torch.Tensor:
    """Đưa tensor ảnh đã Normalize về khoảng [0, 1] để hiển thị."""
    mean = torch.tensor(NORMALIZE_MEAN, dtype=images.dtype).view(-1, 1, 1)
    std = torch.tensor(NORMALIZE_STD, dtype=images.dtype).view(-1, 1, 1)
    return (images * std + mean).clamp(0, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    num_images: int = 16,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lấy một số ảnh đầu tiên, nhãn thật và nhãn model dự đoán.

    Args:
        model: Model PyTorch đã khởi tạo; model có thể đang ở train hoặc eval.
        data_loader: DataLoader trả về từng cặp ``(images, labels)``.
        num_images: Số ảnh tối đa cần lấy.
        device: Device dùng để dự đoán. Nếu không truyền, dùng device của model.

    Returns:
        Tuple ``(images, true_labels, predicted_labels)`` đều đặt trên CPU.

    Raises:
        ValueError: Nếu ``num_images`` không hợp lệ hoặc DataLoader không có ảnh.
    """
    if num_images < 1:
        raise ValueError("num_images phải lớn hơn hoặc bằng 1.")

    model_device = _get_model_device(model, device)
    was_training = model.training
    model.eval()

    image_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    prediction_batches: list[torch.Tensor] = []
    collected = 0

    try:
        for images, labels in data_loader:
            remaining = num_images - collected
            if remaining <= 0:
                break

            images = images[:remaining]
            labels = labels[:remaining]
            logits = model(images.to(model_device))
            predictions = logits.argmax(dim=1)

            image_batches.append(images.detach().cpu())
            label_batches.append(labels.detach().cpu())
            prediction_batches.append(predictions.detach().cpu())
            collected += labels.size(0)
    finally:
        model.train(was_training)

    if collected == 0:
        raise ValueError("data_loader không có ảnh để dự đoán.")

    return (
        torch.cat(image_batches, dim=0),
        torch.cat(label_batches, dim=0),
        torch.cat(prediction_batches, dim=0),
    )


def show_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    num_images: int = 16,
    columns: int = 4,
    device: Optional[torch.device] = None,
    class_names: Sequence[str] = CLASS_NAMES,
    save_path: Optional[PathLike] = None,
    show: bool = False,
) -> Path:
    """Vẽ lưới ảnh FashionMNIST cùng nhãn thật và nhãn dự đoán.

    Tiêu đề mỗi ảnh có màu xanh nếu dự đoán đúng và đỏ nếu dự đoán sai.
    Ảnh mặc định được lưu tại ``results/figures/predictions_grid.png``.

    Args:
        model: Model đã huấn luyện hoặc model cần kiểm tra dự đoán.
        data_loader: Thường là ``test_loader`` do ``get_data_loaders`` tạo ra.
        num_images: Số ảnh tối đa hiển thị.
        columns: Số cột trong lưới ảnh.
        device: Device chạy suy luận; mặc định lấy từ model.
        class_names: Danh sách tên lớp theo thứ tự nhãn.
        save_path: File PNG đầu ra. Nếu không truyền, dùng vị trí mặc định.
        show: Nếu ``True``, hiện biểu đồ sau khi lưu.

    Returns:
        Đường dẫn tuyệt đối của file ảnh đã lưu.
    """
    if columns < 1:
        raise ValueError("columns phải lớn hơn hoặc bằng 1.")

    images, true_labels, predicted_labels = collect_predictions(
        model,
        data_loader,
        num_images=num_images,
        device=device,
    )
    images = _denormalize(images)

    image_count = images.size(0)
    rows = (image_count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)

    for image_index, axis in enumerate(axes.flat):
        if image_index >= image_count:
            axis.axis("off")
            continue

        image = images[image_index]
        if image.size(0) == 1:
            axis.imshow(image.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(image.permute(1, 2, 0).numpy())

        true_label = int(true_labels[image_index].item())
        predicted_label = int(predicted_labels[image_index].item())
        is_correct = true_label == predicted_label
        title_color = "forestgreen" if is_correct else "crimson"

        axis.set_title(
            f"True: {_class_name(true_label, class_names)}\n"
            f"Pred: {_class_name(predicted_label, class_names)}",
            color=title_color,
            fontsize=9,
        )
        axis.axis("off")

    figure.suptitle("FashionMNIST predictions", fontsize=14)
    figure.tight_layout()

    output_path = Path(save_path) if save_path is not None else FIGURES_DIR / "predictions_grid.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(figure)
    return output_path.resolve()


def export_predictions_csv(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    num_images: int = 100,
    device: Optional[torch.device] = None,
    class_names: Sequence[str] = CLASS_NAMES,
    save_path: Optional[PathLike] = None,
) -> Path:
    """Xuất bảng nhãn thật và nhãn dự đoán thành CSV.

    CSV mặc định được lưu tại
    ``data/output/predictions/predictions.csv``. Đây là dữ liệu hỗ trợ phân
    tích các trường hợp dự đoán sai trong phần kết quả thực nghiệm.
    """
    _, true_labels, predicted_labels = collect_predictions(
        model,
        data_loader,
        num_images=num_images,
        device=device,
    )

    output_path = Path(save_path) if save_path is not None else PREDICTIONS_DIR / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_index",
                "true_label_id",
                "true_label",
                "predicted_label_id",
                "predicted_label",
                "correct",
            ],
        )
        writer.writeheader()

        for image_index, (true_label, predicted_label) in enumerate(
            zip(true_labels.tolist(), predicted_labels.tolist())
        ):
            writer.writerow(
                {
                    "image_index": image_index,
                    "true_label_id": true_label,
                    "true_label": _class_name(true_label, class_names),
                    "predicted_label_id": predicted_label,
                    "predicted_label": _class_name(predicted_label, class_names),
                    "correct": true_label == predicted_label,
                }
            )

    return output_path.resolve()
