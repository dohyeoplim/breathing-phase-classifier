import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_freq_bins=177, gru_hidden_size=128, num_gru_layers=2, dropout_rate=0.3):
        super().__init__()

        self.num_freq_bins = num_freq_bins
        self.gru_hidden_size = gru_hidden_size
        self.num_gru_layers = num_gru_layers
        self.dropout_rate = dropout_rate

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)) # 주파수 차원만 절반으로
        )

        # CNN Encoder를 통과한 후의 주파수 빈 수 동적 계산
        # 입력이 (B, 1, F, T) 일 때, 각 MaxPool(kernel_size=(k_h, k_w))는 F를 F // k_h 로 줄임
        # (2,2) -> F // 2
        # (2,2) -> F // 4
        # (2,1) -> F // 8 (주파수 축 kernel_size[0]가 2이므로)
        cnn_output_freq_bins = self.num_freq_bins // 8 
        if cnn_output_freq_bins == 0 : # 매우 작은 num_freq_bins 입력에 대한 방어
            cnn_output_freq_bins = 1
            print(f"Warning: cnn_output_freq_bins became 0 for num_freq_bins={self.num_freq_bins}. Setting to 1.")


        self.gru_input_size = 128 * cnn_output_freq_bins # 128은 마지막 Conv 레이어의 out_channels

        self.temporal_gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_gru_layers > 1 else 0.0 # num_layers=1일 때 dropout=0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.gru_hidden_size * 2, 64), # 양방향 GRU
            nn.ReLU(),
            nn.Dropout(self.dropout_rate / 2.0), # Dropout 비율 조정
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, t, c * f) # GRU 입력: [B, Time, Features]
        x, _ = self.temporal_gru(x)
        x = x.mean(dim=1) # 시간 축에 대한 평균 풀링
        x = self.classifier(x)
        return x