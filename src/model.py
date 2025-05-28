import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1 - Start with fewer filters
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  # Larger kernel for audio
            nn.BatchNorm2d(16, momentum=0.05),  # Lower momentum
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),  # Light dropout early

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.05),
            nn.ReLU(inplace=True),
        )

        self.temporal_gru = nn.GRU(
            input_size=128 * 8,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # 64*2 from bidirectional
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    #
    # def manual_adaptive_pool(self, x, output_size):
    #     b, c, h, w = x.shape
    #     x = F.interpolate(x, size=(output_size, w), mode='bilinear', align_corners=False)
    #     return x

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Adaptive pooling
        b, c, h, w = x.shape
        if h != 8:
            x = F.interpolate(x, size=(8, w), mode='bilinear', align_corners=False)

        # Temporal modeling
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, -1)
        x, _ = self.temporal_gru(x)

        # Global average pooling
        x = x.mean(dim=1)

        return self.classifier(x)