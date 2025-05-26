import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))  # [B, 64, freq', time']
        )

        # GRU input : (batch, time, features)
        self.gru_input_size = 64 * 22  # from 90 → 45 → 22 freq bins

        self.temporal_gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 2, 1)
        )

    def forward(self, x):
        x = self.encoder(x)  # [B, 64, freq, time]
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, time, channels, freq]
        x = x.reshape(b, t, -1)    # [B, time, features]
        x, _ = self.temporal_gru(x)
        x = x.mean(dim=1)
        return self.classifier(x)