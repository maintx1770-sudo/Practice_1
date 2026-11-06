"""So sánh các kiến trúc mạng (MLP, CNN, DeepCNN) trên FashionMNIST.

Chạy từ thư mục gốc dự án:
    python -m experiments.compare_models --skip-train   # chỉ đếm tham số, in kích thước
    python -m experiments.compare_models --epochs 1     # thử nhanh 1 epoch
    python -m experiments.compare_models --epochs 5     # chạy so sánh thật

Điều kiện huấn luyện giống nhau cho mọi mô hình (cùng seed, batch size,
learning rate, optimizer, số epoch) để so sánh công bằng.
"""

import argparse
import time

import pandas as pd
import torch
from torch import nn, optim

from src.config import (
    CHECKPOINT_DIR,
    EPOCHS,
    LEARNING_RATE,
    TABLES_DIR,
    ensure_dirs,
    get_device,
    set_seed,
)
from src.evaluation.metrics import evaluate
from src.models.model_io import save_model
from src.models.neural_network import CNN, MLP, DeepCNN
from src.preprocessing.dataset import get_data_loaders
from src.training.trainer import fit

MODELS = {"MLP": MLP, "CNN": CNN, "DeepCNN": DeepCNN}


def count_parameters(model):
    """Trả về (tổng tham số, tham số có thể huấn luyện)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def trace_shapes(model, input_size=(1, 1, 28, 28)):
    """Cho 1 ảnh giả chạy qua mô hình, ghi lại kích thước đầu ra của từng lớp."""
    shapes = []
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            shapes.append((name, tuple(output.shape)))

        return hook

    for name, layer in model.named_modules():
        if len(list(layer.children())) == 0:  # chỉ lấy các lớp "lá"
            label = f"{name} ({layer.__class__.__name__})"
            hooks.append(layer.register_forward_hook(make_hook(label)))

    model.eval()
    with torch.no_grad():
        model(torch.randn(*input_size))
    for h in hooks:
        h.remove()
    return shapes


def print_architecture_info():
    """In số tham số và kích thước tensor qua từng lớp của mọi mô hình."""
    for name, model_cls in MODELS.items():
        model = model_cls()
        total, trainable = count_parameters(model)
        print(f"\n=== {name} ===")
        print(f"Tổng tham số: {total:,} | Có thể huấn luyện: {trainable:,}")
        for layer_name, shape in trace_shapes(model):
            print(f"  {layer_name:<35} -> {shape}")


def train_and_evaluate(name, model_cls, train_loader, test_loader, epochs, device):
    """Huấn luyện một mô hình, trả về (bảng tóm tắt, history theo epoch)."""
    set_seed()  # cùng seed cho mọi mô hình -> so sánh công bằng
    model = model_cls().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start = time.perf_counter()
    history = fit(
        model,
        train_loader,
        test_loader,
        loss_fn,
        optimizer,
        epochs=epochs,
        device=device,
        verbose=True,
    )
    train_time = time.perf_counter() - start

    final = evaluate(model, test_loader, loss_fn, device)
    total, trainable = count_parameters(model)
    save_model(model, CHECKPOINT_DIR / f"{name.lower()}_compare.pth")

    summary = {
        "model": name,
        "total_params": total,
        "trainable_params": trainable,
        "epochs": epochs,
        "train_time_s": round(train_time, 1),
        "test_loss": round(final["loss"], 4),
        "test_accuracy": round(final["accuracy"], 4),
    }
    return summary, history


def main():
    parser = argparse.ArgumentParser(description="So sánh MLP, CNN, DeepCNN")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Số epoch mỗi mô hình")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Chỉ in số tham số và kích thước tensor, không huấn luyện",
    )
    args = parser.parse_args()

    print_architecture_info()
    if args.skip_train:
        return

    ensure_dirs()
    device = get_device()
    print(f"\nHuấn luyện trên thiết bị: {device}")
    train_loader, test_loader = get_data_loaders()

    summaries, all_history = [], []
    for name, model_cls in MODELS.items():
        print(f"\n##### Huấn luyện {name} #####")
        summary, history = train_and_evaluate(
            name, model_cls, train_loader, test_loader, args.epochs, device
        )
        summaries.append(summary)
        all_history += [{"model": name, **row} for row in history]

    summary_df = pd.DataFrame(summaries)
    print("\n" + summary_df.to_string(index=False))
    summary_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(all_history).to_csv(TABLES_DIR / "model_comparison_history.csv", index=False)
    print(f"\nĐã lưu bảng kết quả vào {TABLES_DIR}")


if __name__ == "__main__":
    main()