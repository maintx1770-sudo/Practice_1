import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import fit, train_one_epoch


def _setup():
    torch.manual_seed(0)
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, loader, nn.CrossEntropyLoss(), opt


def test_one_epoch_runs_with_valid_loss():
    model, loader, loss_fn, opt = _setup()
    res = train_one_epoch(model, loader, loss_fn, opt)
    assert torch.isfinite(torch.tensor(res["loss"]))
    assert 0.0 <= res["accuracy"] <= 1.0


def test_weights_are_updated():
    model, loader, loss_fn, opt = _setup()
    before = [p.detach().clone() for p in model.parameters()]
    train_one_epoch(model, loader, loss_fn, opt)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_fit_history_keys_and_length():
    model, loader, loss_fn, opt = _setup()
    hist = fit(model, loader, loader, loss_fn, opt, epochs=3, verbose=False)
    assert len(hist) == 3
    assert {"epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"} <= set(hist[0])


def test_loss_decreases_on_learnable_data():
    torch.manual_seed(0)
    x = torch.randn(128, 1, 28, 28)
    y = x.flatten(1)[:, :10].argmax(dim=1)  # nhãn học được từ 10 pixel đầu
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    hist = fit(model, loader, None, nn.CrossEntropyLoss(), opt, epochs=8, verbose=False)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
