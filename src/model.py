import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        layers = []
        for i in range(num_convs):
            layers += [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        self.block = nn.Sequential(*layers)
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.proj(x)

class Model(nn.Module):
    def __init__(self, num_scalar_features=11):
        super().__init__()

        self.vgg_blocks = nn.Sequential(
            VGGBlock(1, 64, 2),
            nn.MaxPool2d(2, 2),
            VGGBlock(64, 128, 2),
            nn.MaxPool2d(2, 2),
            VGGBlock(128, 256, 3),
            nn.MaxPool2d(2, 2),
            VGGBlock(256, 512, 3),
            nn.MaxPool2d(2, 2),
            # VGGBlock(512, 512, 3),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.scalar_net = nn.Sequential(
            nn.Linear(num_scalar_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)