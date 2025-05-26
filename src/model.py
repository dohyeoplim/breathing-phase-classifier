import torch
import torch.nn as nn
import torch.nn.functional as functional

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, number_of_channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_mixing = nn.Sequential(
            nn.Linear(number_of_channels, number_of_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(number_of_channels // reduction_ratio, number_of_channels),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        batch_size, channels, _, _ = input_tensor.size()
        squeeze = self.global_pool(input_tensor).view(batch_size, channels)
        excitation = self.channel_mixing(squeeze).view(batch_size, channels, 1, 1)
        return input_tensor * excitation


class ResidualConvolutionalBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout_rate: float = 0.2):
        super().__init__()
        self.convolutional_sequence = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.identity_projection = None
        if input_channels != output_channels:
            self.identity_projection = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.attention_block = SqueezeExcitationBlock(output_channels)

    def forward(self, input_tensor):
        identity = input_tensor if self.identity_projection is None else self.identity_projection(input_tensor)
        output = self.convolutional_sequence(input_tensor)
        output = self.attention_block(output)
        return output + identity


class ResidualNetworkForBreathingAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2)
        )
        self.residual_blocks = nn.Sequential(
            ResidualConvolutionalBlock(16, 32),
            nn.MaxPool2d(kernel_size=2),
            ResidualConvolutionalBlock(32, 64),
            nn.MaxPool2d(kernel_size=2)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, input_tensor):
        x = self.input_block(input_tensor)
        x = self.residual_blocks(x)
        x = self.global_pooling(x)
        x = self.classifier(x)
        return x