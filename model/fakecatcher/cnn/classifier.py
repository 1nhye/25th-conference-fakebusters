from basecnn import baseCNN
from resnet import ResNet34
from effinet import EfficientNetB3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Union, Tuple


class Classifier:
    def __init__(self, model_name: str, w: int) -> None:
        self.model = self.get_model(model_name, w)

    @staticmethod
    def get_model(model_name: str, w: int) -> Union[baseCNN, ResNet34, EfficientNetB3]:
        if model_name == 'baseCNN':
            model = baseCNN(w)
        elif model_name == 'ResNet34':
            model = ResNet34(w)
        elif model_name == 'EfficientNetB3':
            model = EfficientNetB3(w)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        return model

    def train(self, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, num_epochs: int = 10) -> None:
        model = self.model
        model.to(device)
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = (outputs > 0.5).float()
                    correct += (preds == labels).sum().item()
            val_loss /= len(val_loader)
            val_accuracy = correct / len(val_loader.dataset)

            # Print training and validation results
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')

    def test(self, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
        model = self.model
        model.to(device)  # Move the model to the device
        model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)  # Change labels to (batch_size, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = (outputs > 0.5).float()  # Use 0.5 threshold for predictions
                correct += (preds == labels).sum().item()
        accuracy = correct / len(dataloader.dataset)
        return total_loss / len(dataloader), accuracy
    
    def predict(self, input: torch.Tensor, device: torch.device) -> torch.Tensor:
        model = self.model
        model.to(device)
        model.eval()
        return model(input)
    
    def load_model(self, model_path: str) -> nn.Module:
        model = self.model
        model.load_state_dict(torch.load(model_path))
        return model
    
    def save_model(self, model_path: str) -> None:
        model = self.model
        torch.save(model.state_dict(), model_path)
