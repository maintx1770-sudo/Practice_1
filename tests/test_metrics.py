import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import accuracy_from_logits, evaluate, predict_all


def _loader(n=32, batch=8):
    x = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch)


def _model():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def test_accuracy_in_range_and_correct():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    assert accuracy_from_logits(logits, torch.tensor([0, 1, 1])) == pytest.approx(2 / 3)


def test_evaluate_returns_valid_loss_and_accuracy():
    res = evaluate(_model(), _loader())
    assert set(res) == {"loss", "accuracy"}
    assert 0.0 <= res["accuracy"] <= 1.0
    assert torch.isfinite(torch.tensor(res["loss"]))


def test_evaluate_does_not_change_weights_or_mode():
    m = _model()
    m.train()
    before = [p.clone() for p in m.parameters()]
    evaluate(m, _loader())
    assert m.training
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_predict_all_shapes():
    y_true, y_pred = predict_all(_model(), _loader(n=20))
    assert y_true.shape == y_pred.shape == (20,)


def test_accuracy_perfect():
    """TC-16: dự đoán trùng hoàn toàn với nhãn thật -> accuracy = 1.0."""
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    assert accuracy_from_logits(logits, torch.tensor([0, 1, 2])) == pytest.approx(1.0)


def test_accuracy_wrong():
    """TC-17: dự đoán sai hoàn toàn -> accuracy = 0.0."""
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    assert accuracy_from_logits(logits, torch.tensor([1, 2, 0])) == pytest.approx(0.0)
