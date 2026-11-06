from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt

from src.config import FIGURES_DIR


History = Union[Sequence[Mapping[str, Any]], Mapping[str, Sequence[Any]]]
PathLike = Union[str, Path]


def _history_to_columns(history: History) -> dict[str, list[Any]]:
    """Chuyển history dạng list-of-dicts hoặc dict-of-lists thành các cột.

    Hàm hỗ trợ hai định dạng để tiện thử với history giả và dùng lại với kết
    quả thật từ trainer. Tất cả các cột phải có cùng số phần tử.
    """
    if isinstance(history, Mapping):
        if not history:
            raise ValueError("History không được rỗng.")

        columns = {key: list(values) for key, values in history.items()}
        lengths = {len(values) for values in columns.values()}
        if len(lengths) != 1:
            raise ValueError("Các cột trong history phải có cùng số phần tử.")
        if lengths == {0}:
            raise ValueError("History không được rỗng.")
        return columns

    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise TypeError(
            "history phải là list các dictionary hoặc dictionary chứa các list."
        )
    if not history:
        raise ValueError("History không được rỗng.")
    if not all(isinstance(row, Mapping) for row in history):
        raise TypeError("Mỗi phần tử của history phải là một dictionary.")

    keys = set().union(*(row.keys() for row in history))
    return {key: [row.get(key) for row in history] for key in keys}


def _numeric_column(columns: Mapping[str, Sequence[Any]], key: str) -> Optional[list[float]]:
    """Trả về cột số hoặc ``None`` nếu cột không có trong history."""
    values = columns.get(key)
    if values is None:
        return None
    if any(value is None for value in values):
        raise ValueError(f"Cột '{key}' có giá trị bị thiếu.")

    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Cột '{key}' phải chứa các giá trị số.") from error


def plot_loss(
    history: History,
    save_path: Optional[PathLike] = None,
    *,
    show: bool = False,
    title: str = "Training history",
) -> Path:
    """Vẽ và lưu loss curve, kèm accuracy curve nếu history có dữ liệu accuracy.

    Args:
        history: List các dictionary từ ``trainer.fit()`` hoặc dictionary các
            list. Các key loss hợp lệ là ``train_loss`` và ``test_loss`` hoặc
            ``val_loss``. Accuracy là tùy chọn.
        save_path: Đường dẫn ảnh đầu ra. Nếu không truyền, ảnh được lưu là
            ``results/figures/training_history.png``.
        show: Nếu ``True``, hiển thị biểu đồ sau khi lưu. Mặc định là ``False``
            để script chạy được trong môi trường không có giao diện đồ họa.
        title: Tiêu đề chung của biểu đồ.

    Returns:
        Đường dẫn tuyệt đối của file PNG đã lưu.

    Raises:
        ValueError: Nếu history rỗng, thiếu ``train_loss`` hoặc chứa dữ liệu
            không hợp lệ.
    """
    columns = _history_to_columns(history)
    train_loss = _numeric_column(columns, "train_loss")
    if train_loss is None:
        raise ValueError("History phải có cột 'train_loss'.")

    validation_loss = _numeric_column(columns, "val_loss")
    validation_label = "Validation loss"
    if validation_loss is None:
        validation_loss = _numeric_column(columns, "test_loss")
        validation_label = "Test loss"

    train_accuracy = _numeric_column(columns, "train_accuracy")
    validation_accuracy = _numeric_column(columns, "val_accuracy")
    validation_accuracy_label = "Validation accuracy"
    if validation_accuracy is None:
        validation_accuracy = _numeric_column(columns, "test_accuracy")
        validation_accuracy_label = "Test accuracy"

    epochs = _numeric_column(columns, "epoch")
    if epochs is None:
        epochs = list(range(1, len(train_loss) + 1))

    if len(epochs) != len(train_loss):
        raise ValueError("Cột 'epoch' và 'train_loss' phải có cùng số phần tử.")

    has_accuracy = train_accuracy is not None or validation_accuracy is not None
    number_of_plots = 2 if has_accuracy else 1
    figure, axes = plt.subplots(1, number_of_plots, figsize=(6 * number_of_plots, 4.5))
    if number_of_plots == 1:
        axes = [axes]

    loss_axis = axes[0]
    loss_axis.plot(epochs, train_loss, marker="o", linewidth=2, label="Train loss")
    if validation_loss is not None:
        loss_axis.plot(
            epochs,
            validation_loss,
            marker="o",
            linestyle="--",
            linewidth=2,
            label=validation_label,
        )
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Loss curve")
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend()

    if has_accuracy:
        accuracy_axis = axes[1]
        if train_accuracy is not None:
            accuracy_axis.plot(
                epochs,
                train_accuracy,
                marker="o",
                linewidth=2,
                label="Train accuracy",
            )
        if validation_accuracy is not None:
            accuracy_axis.plot(
                epochs,
                validation_accuracy,
                marker="o",
                linestyle="--",
                linewidth=2,
                label=validation_accuracy_label,
            )
        accuracy_axis.set_xlabel("Epoch")
        accuracy_axis.set_ylabel("Accuracy")
        accuracy_axis.set_title("Accuracy curve")
        accuracy_axis.set_ylim(0, 1)
        accuracy_axis.grid(True, alpha=0.3)
        accuracy_axis.legend()

    figure.suptitle(title)
    figure.tight_layout()

    output_path = Path(save_path) if save_path is not None else FIGURES_DIR / "training_history.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(figure)
    return output_path.resolve()


def plot_training_history(
    history: History,
    save_path: Optional[PathLike] = None,
    *,
    show: bool = False,
    title: str = "Training history",
) -> Path:
    """Tên hàm đầy đủ, là alias của :func:`plot_loss`."""
    return plot_loss(history, save_path, show=show, title=title)


DEMO_HISTORY = [
    {
        "epoch": 1,
        "train_loss": 1.45,
        "test_loss": 1.12,
        "train_accuracy": 0.49,
        "test_accuracy": 0.60,
    },
    {
        "epoch": 2,
        "train_loss": 0.93,
        "test_loss": 0.80,
        "train_accuracy": 0.68,
        "test_accuracy": 0.72,
    },
    {
        "epoch": 3,
        "train_loss": 0.72,
        "test_loss": 0.68,
        "train_accuracy": 0.75,
        "test_accuracy": 0.77,
    },
    {
        "epoch": 4,
        "train_loss": 0.60,
        "test_loss": 0.61,
        "train_accuracy": 0.80,
        "test_accuracy": 0.79,
    },
    {
        "epoch": 5,
        "train_loss": 0.52,
        "test_loss": 0.57,
        "train_accuracy": 0.83,
        "test_accuracy": 0.81,
    },
]


if __name__ == "__main__":
    saved_file = plot_loss(
        DEMO_HISTORY,
        FIGURES_DIR / "plot_loss_demo.png",
        title="Demo history (synthetic data)",
    )
    print("Saved demo plot.")
