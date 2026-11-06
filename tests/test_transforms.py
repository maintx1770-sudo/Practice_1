import pytest
import torch
from PIL import Image
from src.preprocessing.transforms import get_test_transforms, get_train_transforms

def test_to_tensor(sample_pil_image):
    """TC-01: Kiểm tra train transform chuyển PIL Image thành Tensor float32 kích thước (1, 28, 28)."""
    transform = get_train_transforms()
    tensor_img = transform(sample_pil_image)

    assert isinstance(tensor_img, torch.Tensor)
    assert tensor_img.shape == (1, 28, 28)
    assert tensor_img.dtype == torch.float32

def test_normalize(sample_pil_image):
    """TC-02: Kiểm tra chuẩn hóa pixel về khoảng [-1, 1]."""
    transform = get_train_transforms()
    tensor_img = transform(sample_pil_image)

    assert tensor_img.shape == (1, 28, 28)
    assert torch.all(tensor_img >= -1.0) and torch.all(tensor_img <= 1.0)

def test_transform_consistency(sample_pil_image):
    """TC-03: Kiểm tra tính nhất quán của test transform."""
    transform = get_test_transforms()
    tensor_img = transform(sample_pil_image)

    assert isinstance(tensor_img, torch.Tensor)
    assert tensor_img.shape == (1, 28, 28)
    assert tensor_img.dtype == torch.float32