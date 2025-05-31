import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2, use_se=False):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self.use_se = use_se
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        return self.se(x)

class Model(nn.Module):
    def __init__(self, num_scalar_features=8, use_se=True):
        super().__init__()

        self.vgg_blocks = nn.Sequential(
            VGGBlock(1, 64, 2, use_se),
            nn.MaxPool2d(2, 2),
            VGGBlock(64, 128, 2, use_se),
            nn.MaxPool2d(2, 2),
            VGGBlock(128, 256, 3, use_se),
            nn.MaxPool2d(2, 2),
            VGGBlock(256, 512, 3, use_se),
            nn.MaxPool2d(2, 2),
            VGGBlock(512, 512, 3, use_se),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.scalar_net = nn.Sequential(
            nn.Linear(num_scalar_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        self._initialize_weights()

    def forward(self, features, scalars):
        if features is not None:
            x = self.vgg_blocks(features)
            x = x.view(x.size(0), -1)
        else:
            x = torch.zeros((scalars.size(0), 512), device=scalars.device)

        s = self.scalar_net(scalars)
        combined = torch.cat([x, s], dim=1)
        fused = self.fusion(combined)
        return self.classifier(fused)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)