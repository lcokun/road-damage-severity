from __future__ import annotations
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, img_size: int, n_classes: int):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),    # 112
            block(32, 64),   # 56
            block(64, 128),  # 28
            block(128, 128), # 14
            nn.Dropout2d(0.1),
        )

        feat_hw = img_size // 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feat_hw * feat_hw, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))