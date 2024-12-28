from basecnn import baseCNN
from resnet import ResNet34
from effinet import EfficientNetB3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model:
    @staticmethod
    def get_model(model_name, w):
        if model_name == 'baseCNN':
            return baseCNN(w)
        elif model_name == 'ResNet34':
            return ResNet34(w)
        elif model_name == 'EfficientNetB3':
            return EfficientNetB3(w)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @staticmethod
    def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
        model.to(device)
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(-1, 1).float()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

            val_loss, val_acurracy = Model.test(model, val_loader, device)

    @staticmethod
    def test(model, val_loader, device):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.BCELoss()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(-1, 1).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                # Accuracy 계산
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        accuracy = correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        return val_loss, accuracy