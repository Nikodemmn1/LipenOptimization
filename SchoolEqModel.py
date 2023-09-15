import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub


''' Model for the base 6 or 3 classes  without any fancy things'''
class SchoolEqModel(nn.Module):
    def __init__(self, num_classes):
        super(SchoolEqModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten()
        )

        self.classification_head = nn.Sequential(
            nn.Linear(2048, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
            DeQuantStub(),
            nn.Softmax(dim=-1)
        )

        self.quant_stub = QuantStub()

    def forward(self, x):
        x = self.quant_stub(x)
        x = self.feature_extractor(x)
        x = self.classification_head(x)
        return x
