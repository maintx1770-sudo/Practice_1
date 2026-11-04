import pytest
import torch

from src.config import NUM_CLASSES
from src.models.neural_network import CNN, MLP, DeepCNN

MODELS = [MLP, CNN, DeepCNN]


@pytest.mark.parametrize("model_cls", MODELS)
def test_output_shape(model_cls):
    model = model_cls()
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    assert out.shape == (8, NUM_CLASSES)


@pytest.mark.parametrize("model_cls", MODELS)
def test_output_is_finite(model_cls):
    model = model_cls()
    out = model(torch.randn(4, 1, 28, 28))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("model_cls", MODELS)
def test_single_image(model_cls):
    model = model_cls().eval()
    out = model(torch.randn(1, 1, 28, 28))
    assert out.shape == (1, NUM_CLASSES)


@pytest.mark.parametrize("model_cls", MODELS)
def test_backward_runs(model_cls):
    model = model_cls()
    out = model(torch.randn(4, 1, 28, 28))
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())

