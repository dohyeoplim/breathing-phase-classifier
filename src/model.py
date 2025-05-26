import torch
import torch.nn as nn
import torch.nn.functional as functional

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, number_of_channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_mixing = nn.Sequential(
            nn.Linear(number_of_channels, number_of_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(number_of_channels // reduction_ratio, number_of_channels),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        batch_size, channels, _, _ = input_tensor.size()
        squeeze = self.global_pool(input_tensor).view(batch_size, channels)
        excitation = self.channel_mixing(squeeze).view(batch_size, channels, 1, 1)
        return input_tensor * excitation


class ResidualConvolutionalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2):
        super().__init__()
        self.use_projection = in_channels != out_channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        self.attention = SqueezeExcitationBlock(out_channels)

        if self.use_projection:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block(x)
        out = self.attention(out)
        return out + identity


class ResidualNetworkForBreathingAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.residual_layers = nn.Sequential(
            ResidualConvolutionalBlock(16, 32, dropout_rate=0.2),
            nn.MaxPool2d(kernel_size=2),
            ResidualConvolutionalBlock(32, 64, dropout_rate=0.2),
            nn.MaxPool2d(kernel_size=2)
        )

        self.temporal_gru = nn.GRU(
            input_size=64, hidden_size=64,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.residual_layers(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        x, _ = self.temporal_gru(x)
        x = x.mean(dim=1)
        return self.classifier(x)