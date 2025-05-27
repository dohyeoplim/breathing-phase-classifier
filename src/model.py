import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.temporal_gru = nn.GRU(
            input_size=128 * 8,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def manual_adaptive_pool(self, x, output_size):
        b, c, h, w = x.shape
        x = F.interpolate(x, size=(output_size, w), mode='bilinear', align_corners=False)
        return x


    def forward(self, x):
        x = self.encoder(x)  # [B, 64, freq, time]
        x = self.manual_adaptive_pool(x, output_size=8)  # [B, 128, 8, time']

        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, time, channels, freq]
        x = x.reshape(b, t, -1)    # [B, time, features]
        x, _ = self.temporal_gru(x)
        x = x.mean(dim=1)
        return self.classifier(x)