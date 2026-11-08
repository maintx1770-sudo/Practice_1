"""Thử nghiệm hyperparameters cho FashionMNIST.

Chạy từ thư mục gốc project, ví dụ:

    python -m experiments.tune_hyperparams --quick --epochs 3

Chế độ mặc định chạy toàn bộ kế hoạch thử nghiệm. ``--quick`` chỉ chạy ba
learning rate để kiểm tra pipeline hoặc chạy nhanh trên máy CPU.

Tuning chọn cấu hình dựa trên validation set tách từ tập train. Tập test chỉ
được đánh giá một lần sau khi đã chọn được cấu hình tốt nhất.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EPOCHS,
    FIGURES_DIR,
    NUM_WORKERS,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    TABLES_DIR,
    ensure_dirs,
    get_device,
    set_seed,
)
from src.evaluation.metrics import evaluate
from src.models.model_io import save_model
from src.models.neural_network import CNN, DeepCNN, MLP
from src.preprocessing.dataset import get_fashion_mnist_datasets
from src.training.trainer import fit
from src.visualization.plot_loss import plot_loss
from src.visualization.show_predictions import export_predictions_csv, show_predictions


@dataclass(frozen=True)
class ExperimentConfig:
    """Cấu hình của một lần thử nghiệm có kiểm soát."""

    experiment_id: str
    purpose: str
    model_name: str = "cnn"
    learning_rate: float = 1e-3
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    optimizer_name: str = "adam"
    dropout: float = 0.25
    weight_decay: float = 0.0
    momentum: float = 0.9


def full_experiments(base_epochs: int = EPOCHS) -> list[ExperimentConfig]:
    """Trả về kế hoạch tối thiểu: LR, batch size, epoch, optimizer, dropout.

    Mỗi nhóm chỉ thay đổi một yếu tố so với baseline ``TUNE-BASE``. Những cấu
    hình này có thể mất nhiều thời gian trên CPU; dùng ``--quick`` để kiểm tra
    pipeline trước.
    """
    return [
        ExperimentConfig("TUNE-BASE", "Baseline CNN + Adam", epochs=base_epochs),
        ExperimentConfig("TUNE-LR-LOW", "Learning rate thấp", learning_rate=3e-4, epochs=base_epochs),
        ExperimentConfig("TUNE-LR-HIGH", "Learning rate cao", learning_rate=3e-3, epochs=base_epochs),
        ExperimentConfig("TUNE-BATCH-32", "Batch size nhỏ", batch_size=32, epochs=base_epochs),
        ExperimentConfig("TUNE-BATCH-128", "Batch size lớn", batch_size=128, epochs=base_epochs),
        ExperimentConfig(
            "TUNE-EPOCH-SHORT",
            "Ít epoch hơn baseline",
            epochs=max(1, base_epochs // 2),
        ),
        ExperimentConfig(
            "TUNE-EPOCH-LONG",
            "Nhiều epoch hơn baseline",
            epochs=max(base_epochs + 2, base_epochs * 2),
        ),
        ExperimentConfig(
            "TUNE-OPT-SGD",
            "So sánh SGD momentum với Adam",
            learning_rate=1e-2,
            optimizer_name="sgd",
            epochs=base_epochs,
        ),
        ExperimentConfig("TUNE-DROP-0", "Không dùng dropout", dropout=0.0, epochs=base_epochs),
        ExperimentConfig("TUNE-DROP-20", "Dropout nhẹ", dropout=0.20, epochs=base_epochs),
        ExperimentConfig("TUNE-DROP-50", "Dropout mạnh", dropout=0.50, epochs=base_epochs),
    ]


def quick_experiments(epochs: int = 3) -> list[ExperimentConfig]:
    """Ba learning rate để chạy nhanh và kiểm tra toàn bộ pipeline."""
    return [
        ExperimentConfig(
            "TUNE-QUICK-LR-LOW",
            "Quick run: learning rate 3e-4",
            learning_rate=3e-4,
            epochs=epochs,
        ),
        ExperimentConfig(
            "TUNE-QUICK-LR-BASE",
            "Quick run: learning rate 1e-3",
            learning_rate=1e-3,
            epochs=epochs,
        ),
        ExperimentConfig(
            "TUNE-QUICK-LR-HIGH",
            "Quick run: learning rate 3e-3",
            learning_rate=3e-3,
            epochs=epochs,
        ),
    ]


def build_model(config: ExperimentConfig) -> nn.Module:
    """Khởi tạo model tương ứng với cấu hình thí nghiệm."""
    model_name = config.model_name.lower()
    if model_name == "mlp":
        return MLP()
    if model_name == "cnn":
        return CNN(dropout=config.dropout)
    if model_name in {"deepcnn", "deep_cnn"}:
        return DeepCNN(dropout=config.dropout)
    raise ValueError(f"model_name không hợp lệ: {config.model_name}")


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    """Khởi tạo optimizer đúng theo config."""
    optimizer_name = config.optimizer_name.lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"optimizer_name không hợp lệ: {config.optimizer_name}")


def create_data_splits(
    validation_fraction: float = 0.1,
    seed: int = RANDOM_SEED,
) -> tuple[Dataset, Dataset, Dataset]:
    """Tải FashionMNIST và tách train/validation với seed cố định."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction phải nằm trong khoảng (0, 1).")

    train_dataset, test_dataset = get_fashion_mnist_datasets()
    validation_size = int(len(train_dataset) * validation_fraction)
    train_size = len(train_dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, validation_subset = random_split(
        train_dataset,
        [train_size, validation_size],
        generator=generator,
    )
    return train_subset, validation_subset, test_dataset


def create_loaders(
    train_subset: Dataset,
    validation_subset: Dataset,
    test_dataset: Dataset,
    *,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Tạo DataLoader với cùng train/validation split cho mọi experiment."""
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=shuffle_generator,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return train_loader, validation_loader, test_loader


def _rename_validation_keys(history: list[dict[str, float]]) -> list[dict[str, float]]:
    """Đổi tên key test_* từ ``fit`` thành val_* vì loader truyền vào là validation."""
    converted_history: list[dict[str, float]] = []
    for row in history:
        converted = dict(row)
        if "test_loss" in converted:
            converted["val_loss"] = converted.pop("test_loss")
        if "test_accuracy" in converted:
            converted["val_accuracy"] = converted.pop("test_accuracy")
        converted_history.append(converted)
    return converted_history


def _parameter_count(model: nn.Module) -> int:
    """Đếm tổng số parameter có thể train."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _best_epoch(history: list[dict[str, float]]) -> dict[str, float]:
    """Chọn epoch có validation accuracy cao nhất, hòa thì validation loss thấp hơn."""
    return max(history, key=lambda row: (row["val_accuracy"], -row["val_loss"]))


def run_experiment(
    config: ExperimentConfig,
    train_subset: Dataset,
    validation_subset: Dataset,
    test_dataset: Dataset,
    *,
    device: torch.device,
    seed: int = RANDOM_SEED,
) -> tuple[dict[str, object], list[dict[str, float]], nn.Module, DataLoader]:
    """Train một cấu hình và lưu history/biểu đồ riêng cho cấu hình đó."""
    set_seed(seed)
    train_loader, validation_loader, test_loader = create_loaders(
        train_subset,
        validation_subset,
        test_dataset,
        batch_size=config.batch_size,
        seed=seed,
    )

    model = build_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    start_time = time.perf_counter()
    history: list[dict[str, float]] = []
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_validation_score: Optional[tuple[float, float]] = None

    # Gọi fit từng epoch để giữ state_dict đúng tại epoch validation tốt nhất.
    # Việc này vẫn dùng toàn bộ train/evaluation loop chuẩn trong trainer.py.
    for epoch in range(1, config.epochs + 1):
        epoch_history = _rename_validation_keys(
            fit(
                model,
                train_loader,
                validation_loader,
                loss_fn,
                optimizer,
                epochs=1,
                device=device,
                verbose=False,
            )
        )
        row = epoch_history[0]
        row["epoch"] = epoch
        history.append(row)

        current_score = (row["val_accuracy"], -row["val_loss"])
        if best_validation_score is None or current_score > best_validation_score:
            best_validation_score = current_score
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )

        print(
            f"{config.experiment_id} | epoch {epoch}/{config.epochs} | "
            f"train_loss={row['train_loss']:.4f} | val_loss={row['val_loss']:.4f} | "
            f"val_accuracy={row['val_accuracy']:.4f}"
        )
    duration_seconds = time.perf_counter() - start_time
    assert best_state is not None
    model.load_state_dict(best_state)

    history_path = TABLES_DIR / f"{config.experiment_id.lower()}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    figure_path = plot_loss(
        history,
        FIGURES_DIR / f"{config.experiment_id.lower()}_history.png",
        title=f"{config.experiment_id}: {config.model_name.upper()}",
    )

    best = _best_epoch(history)
    final = history[-1]
    record: dict[str, object] = {
        **asdict(config),
        "seed": seed,
        "device": str(device),
        "train_size": len(train_subset),
        "validation_size": len(validation_subset),
        "test_size": len(test_dataset),
        "parameter_count": _parameter_count(model),
        "duration_seconds": round(duration_seconds, 2),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": best["val_loss"],
        "best_val_accuracy": best["val_accuracy"],
        "final_train_loss": final["train_loss"],
        "final_train_accuracy": final["train_accuracy"],
        "final_val_loss": final["val_loss"],
        "final_val_accuracy": final["val_accuracy"],
        "history_file": str(history_path.relative_to(TABLES_DIR.parent.parent)),
        "figure_file": str(figure_path.relative_to(FIGURES_DIR.parent.parent)),
    }
    return record, history, model, test_loader


def plot_comparison(results: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Vẽ biểu đồ so sánh best validation accuracy giữa các experiment."""
    if results.empty:
        raise ValueError("Không có kết quả để vẽ biểu đồ so sánh.")

    output_path = save_path or FIGURES_DIR / "hyperparameter_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(max(8, len(results) * 1.1), 5))
    bars = axis.bar(results["experiment_id"], results["best_val_accuracy"], color="steelblue")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Best validation accuracy")
    axis.set_title("Hyperparameter tuning comparison")
    axis.tick_params(axis="x", rotation=40)
    axis.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, results["best_val_accuracy"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.01,
            f"{float(value):.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path.resolve()


def run_tuning(
    experiments: Sequence[ExperimentConfig],
    *,
    validation_fraction: float = 0.1,
    seed: int = RANDOM_SEED,
    create_prediction_outputs: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Chạy các experiment, chọn best theo validation và đánh giá một lần trên test.

    Returns:
        ``(results_dataframe, best_summary)``. Các file CSV, PNG và checkpoint
        cũng được lưu vào thư mục output của project.
    """
    if not experiments:
        raise ValueError("Cần ít nhất một ExperimentConfig.")

    ensure_dirs()
    set_seed(seed)
    device = get_device()
    train_subset, validation_subset, test_dataset = create_data_splits(
        validation_fraction=validation_fraction,
        seed=seed,
    )

    records: list[dict[str, object]] = []
    best_config: Optional[ExperimentConfig] = None
    best_record: Optional[dict[str, object]] = None
    best_state: Optional[dict[str, torch.Tensor]] = None

    for index, config in enumerate(experiments, start=1):
        print(f"[{index}/{len(experiments)}] Running {config.experiment_id}")
        record, _, model, _ = run_experiment(
            config,
            train_subset,
            validation_subset,
            test_dataset,
            device=device,
            seed=seed,
        )
        records.append(record)

        is_better = best_record is None or (
            float(record["best_val_accuracy"]),
            -float(record["best_val_loss"]),
        ) > (
            float(best_record["best_val_accuracy"]),
            -float(best_record["best_val_loss"]),
        )
        if is_better:
            best_config = config
            best_record = record
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )

    results = pd.DataFrame(records).sort_values(
        ["best_val_accuracy", "best_val_loss"],
        ascending=[False, True],
    )
    results_path = TABLES_DIR / "hyperparameter_results.csv"
    results.to_csv(results_path, index=False)
    comparison_path = plot_comparison(results)

    assert best_config is not None and best_record is not None and best_state is not None
    best_model = build_model(best_config).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    _, _, test_loader = create_loaders(
        train_subset,
        validation_subset,
        test_dataset,
        batch_size=best_config.batch_size,
        seed=seed,
    )
    test_metrics = evaluate(best_model, test_loader, nn.CrossEntropyLoss(), device)
    checkpoint_path = save_model(best_model, CHECKPOINT_DIR / "best_tuned_model.pth")

    best_summary: dict[str, object] = {
        **best_record,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "checkpoint_file": str(checkpoint_path.relative_to(CHECKPOINT_DIR.parent.parent.parent)),
        "comparison_figure_file": str(comparison_path.relative_to(FIGURES_DIR.parent.parent)),
    }

    if create_prediction_outputs:
        prediction_figure = show_predictions(
            best_model,
            test_loader,
            device=device,
            num_images=16,
            save_path=FIGURES_DIR / "best_tuned_predictions.png",
        )
        prediction_table = export_predictions_csv(
            best_model,
            test_loader,
            device=device,
            num_images=len(test_dataset),
            save_path=PREDICTIONS_DIR / "best_tuned_predictions.csv",
        )
        best_summary["prediction_figure_file"] = str(
            prediction_figure.relative_to(FIGURES_DIR.parent.parent)
        )
        best_summary["prediction_table_file"] = str(
            prediction_table.relative_to(PREDICTIONS_DIR.parent.parent.parent)
        )

    summary_path = TABLES_DIR / "best_hyperparameter_summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(best_summary, summary_file, ensure_ascii=False, indent=2)

    print(f"Best experiment: {best_config.experiment_id}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Results table: {results_path}")
    return results, best_summary


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Tune hyperparameters for FashionMNIST.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run three learning rates instead of the full plan.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Baseline epoch count; quick mode defaults to 3.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of the train set reserved for validation; default 0.1.",
    )
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Do not export prediction image or table for the best model.",
    )
    return parser.parse_args()


def main() -> None:
    """Điểm chạy của module."""
    args = parse_args()
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs phải lớn hơn hoặc bằng 1.")

    if args.quick:
        experiments = quick_experiments(epochs=args.epochs or 3)
    else:
        experiments = full_experiments(base_epochs=args.epochs or EPOCHS)

    run_tuning(
        experiments,
        validation_fraction=args.validation_fraction,
        create_prediction_outputs=not args.no_predictions,
    )


if __name__ == "__main__":
    main()
