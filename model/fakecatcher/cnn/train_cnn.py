import torch
from torch.utils.data import DataLoader, random_split
from cnn.classifier import Model
import yaml
import json
import numpy as np
from torch.utils.data import Dataset


# 설정
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 준비
with open('/root/25th-conference-fakebusters/model/fakecatcher/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    fps_standard = config.get('fps_standard', 30)  # 기본값 30
    time_interval = config.get('seg_time_interval', 1)  # 기본값 1
    w = fps_standard * time_interval

with open('/root/25th-conference-fakebusters/model/fakecatcher/ppg_map_results_updated.json', 'r') as file:
    data = json.load(file)

# Extract PPG Maps and Labels
ppg_maps = []
labels = []
for item in data:
    ppg_maps.append(np.array(item['ppg_map']))  # (64, w) 형태의 배열
    labels.append(item['label'])               # 0 or 1

# Convert to numpy arrays
ppg_maps = np.array(ppg_maps)
labels = np.array(labels)     

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        Args:
            data (np.array): Shape (2831, 64, 90)
            labels (np.array): Shape (2831,)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (data_sample, label)
        """
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(ppg_maps, labels)
print(f"Dataset size: {len(dataset)}")
print(f"Sample PPG map shape: {dataset[0][0].shape}")
print(f"Sample label: {dataset[0][1]}")
train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 준비
model_name = 'ResNet34'  # "ResNet34", "EfficientNetB3"로 변경 가능
model = Model.get_model(model_name, w)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습
Model.train(model, train_loader, val_loader, criterion, optimizer, device, epochs=NUM_EPOCHS)

# # 테스트
# test_loss, test_accuracy = Model.test(model, val_loader, criterion, device)
# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
