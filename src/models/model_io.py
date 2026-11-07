from pathlib import Path

import torch

from src.config import (
    CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_NAME,
    ensure_dirs,
    get_device,
)


def save_model(model, path=None):
    """Lưu trọng số mô hình, trả về đường dẫn đã lưu."""
    ensure_dirs()
    path = Path(path) if path else CHECKPOINT_DIR / DEFAULT_CHECKPOINT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_model(model, path=None, device=None):
    """Nạp trọng số vào `model` (đã khởi tạo đúng kiến trúc), chuyển sag eval()."""
    path = Path(path) if path else CHECKPOINT_DIR / DEFAULT_CHECKPOINT_NAME
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    device = device or get_device()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
