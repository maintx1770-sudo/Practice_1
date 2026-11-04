import pytest
import torch

from src.models.model_io import load_model, save_model
from src.models.neural_network import CNN, MLP, DeepCNN

CPU = torch.device("cpu")


@pytest.mark.parametrize("model_cls", [MLP, CNN, DeepCNN])
def test_save_then_load_same_predictions(model_cls, tmp_path):
    original = model_cls().eval()
    x = torch.randn(4, 1, 28, 28)
    expected = original(x)

    path = save_model(original, tmp_path / "model.pth")
    assert path.exists()

    loaded = load_model(model_cls(), path, device=CPU)  # mô hình mới, trọng số ngẫu nhiên khác
    actual = loaded(x)

    assert torch.allclose(expected, actual, atol=1e-6)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(CNN(), tmp_path / "khong_ton_tai.pth", device=CPU)