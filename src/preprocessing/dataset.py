"""
Module nạp bộ dữ liệu FashionMNIST và khởi tạo DataLoader.
"""
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from src.config import BATCH_SIZE, DATA_DIR, NUM_WORKERS
from src.preprocessing.transforms import get_test_transforms, get_train_transforms

def get_fashion_mnist_datasets(data_dir=None):
    """
    Tải (nếu chưa có) và trả về FashionMNIST train set và test set.
    """
    if data_dir is None:
        data_dir = DATA_DIR / "input"

    train_dataset = FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=get_train_transforms(),
    )

    test_dataset = FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=get_test_transforms(),
    )

    return train_dataset, test_dataset

def get_data_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, data_dir=None):
    """
    Khởi tạo và trả về (train_loader, test_loader).
    """
    train_dataset, test_dataset = get_fashion_mnist_datasets(data_dir=data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader