import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class baseCNN(nn.Module):
    def __init__(self, w):
        super(baseCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer는 Forward Pass 중 Feature Map 크기를 기반으로 설정
        self.fc1 = None
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 64, 90) -> (batch_size, 1, 64, 90)
        x = F.relu(self.conv1(x))  # Conv1
        x = self.pool(x)           # MaxPool1
        x = F.relu(self.conv2(x))  # Conv2
        x = self.pool(x)           # MaxPool2
        x = F.relu(self.conv3(x))  # Conv3
        x = self.pool(x)           # MaxPool3

        # Fully Connected Layer 동적 초기화
        if self.fc1 is None:
            feature_map_size = x.shape[1] * x.shape[2] * x.shape[3]  # Channel * Height * Width
            self.fc1 = nn.Linear(feature_map_size, 64).to(x.device)

        x = torch.flatten(x, start_dim=1)  # Flatten: (batch_size, channels * height * width)
        x = F.relu(self.fc1(x))            # FC1
        x = self.dropout(x)                # Dropout
        x = torch.sigmoid(self.fc2(x))     # FC2 + Sigmoid
        return x
