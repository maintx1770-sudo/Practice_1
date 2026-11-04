"""
Module định nghĩa các phép biến đổi ảnh (transforms) cho FashionMNIST.
"""

from torchvision import transforms
from src.config import NORMALIZE_MEAN, NORMALIZE_STD


def get_train_transforms():
    """
    Trả về chuỗi biến đổi ảnh dành cho tập Train.
    - ToTensor(): Chuyển ảnh PIL thành PyTorch Tensor (giá trị pixel trong [0, 1]).
    - Normalize: Chuẩn hóa Tensor về khoảng [-1, 1] sử dụng mean và std từ config.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


def get_test_transforms():
    """
    Trả về chuỗi biến đổi ảnh dành cho tập kiểm thử/đánh giá.
    Giữ nguyên quy trình chuẩn hóa nhất quán như tập Train, không áp dụng biến đổi ngẫu nhiên.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])