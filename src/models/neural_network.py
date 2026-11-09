import torch
from torch import nn

from src.config import IMAGE_SIZE, NUM_CHANNELS, NUM_CLASSES

class MLP(nn.Module):
      def __init__(self):
            super().__init__()
            in_features = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS # 1*28*28 = 784
            self.model = nn.Sequential(
                  nn.Flatten(),
                  nn.Linear(in_features, 256),
                  nn.ReLU(),
                  nn.Linear(256, 128),
                  nn.ReLU(),
                  nn.Linear(128, NUM_CLASSES)   
            )
            
      def forward(self, x):
            return self.model(x)
      
class CNN(nn.Module):
      def __init__(self, dropout=0.25):
        super().__init__()

        # Phần 1: trích xuất đặc trưng (vẫn là ảnh, chx phẳng)
        self.features = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),  # (B,32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # (B,32,14,14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),            # (B,64,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # (B,64,7,7)
        )

        # Phần 2: phân loại (đã duỗi phẳng thành vector)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                           # (B,3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),                            # (B,10) logits
        )

      def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
  
class DeepCNN(nn.Module):
      def __init__(self, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),  # (B,32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # (B,32,14,14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),            # (B,64,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # (B,64,7,7)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),           # (B,128,7,7)
            nn.ReLU(),
            nn.MaxPool2d(2),                                        # (B,128,3,3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                           # (B,1152)
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_CLASSES),
        )

      def forward(self, x):
        return self.classifier(self.features(x))
