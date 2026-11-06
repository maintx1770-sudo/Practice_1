import pytest
import torch
from torch.utils.data import DataLoader

from src.preprocessing.dataset import (
    get_data_loaders,
    get_fashion_mnist_datasets,
)

def test_mock_dataset(sample_dataset):
    """TC-04: Đọc mock dataset từ conftest (8 mẫu, ảnh (1, 28, 28), nhãn 0-9)."""
    assert len(sample_dataset) == 8
    img, label = sample_dataset[0]
    assert img.shape == (1, 28, 28)
    assert 0 <= label <= 9

def test_mock_loader_batch(sample_loader):
    """TC-05: Lấy batch từ sample_loader (batch=4, ảnh (4, 1, 28, 28), nhãn (4,), torch.long)."""
    images, labels = next(iter(sample_loader))
    assert images.shape == (4, 1, 28, 28)
    assert labels.shape == (4,)
    assert labels.dtype == torch.long

def test_fashionmnist_split_sizes():
    """TC-06: Tải/đọc train và test FashionMNIST thật (Train 60.000, Test 10.000)."""
    train_ds, test_ds = get_fashion_mnist_datasets()
    assert len(train_ds) == 60000
    assert len(test_ds) == 10000

def test_fashionmnist_sample():
    """TC-07: Lấy 1 ảnh từ tập train thật xem đúng tensor (1, 28, 28) và nhãn 0-9."""
    train_ds, _ = get_fashion_mnist_datasets()
    img, label = train_ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 28, 28)
    assert 0 <= label <= 9

def test_get_data_loaders_batch_shapes():
    """Kiểm tra get_data_loaders trả về DataLoader chuẩn."""
    train_loader, test_loader = get_data_loaders(batch_size=16)
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    images, labels = next(iter(train_loader))
    assert images.shape == (16, 1, 28, 28)
    assert labels.shape == (16,)
    assert labels.dtype == torch.long