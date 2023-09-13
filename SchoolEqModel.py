import torch.nn as nn


class SchoolEqModel(nn.Module):
    def __init__(self):
        super(SchoolEqModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten()
        )

        self.classification_head = nn.Sequential(
            nn.Linear(2048, 100),
            nn.Linear(100, 6),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classification_head(x)
        return x
