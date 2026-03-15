import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


class DigitsDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        if self.transform:
            X = self.transform(X)
        return X, y


class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=32, out_features=10)
        )
    
    def forward(self, X):
        X = self.features(X)
        X = self.flatten(X)
        X = self.classifier(X)
        return X


class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4*4*32, out_features=10)
        )
    
    def forward(self, X):
        X = self.features(X)
        X = self.flatten(X)
        X = self.classifier(X)
        return X


def get_transform():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )
    return transform


def train(model, criterion, optimizer, history, num_epochs, train_loader, valid_loader, device, early_stopping):
    base_size = len(history)
    best_loss = torch.inf
    best_params = {}
    no_improve = 0

    for epoch in range(base_size, base_size+num_epochs):

        n_train, n_valid = 0, 0
        running_train_loss, running_valid_loss = 0, 0
        running_train_acc, running_valid_acc = 0, 0

        model.train()
        for X_tr, y_tr in train_loader:
            X_tr, y_tr = X_tr.to(device), y_tr.to(device)
            train_batch_size = len(y_tr)
            n_train += train_batch_size

            optimizer.zero_grad()
            output_tr = model(X_tr)
            loss_tr = criterion(output_tr, y_tr)
            loss_tr.backward()
            optimizer.step()

            pred_tr = torch.argmax(output_tr, dim=-1)
            running_train_loss += loss_tr.item() * train_batch_size
            running_train_acc += (pred_tr == y_tr).sum().item()

        model.eval()
        with torch.no_grad():
            for X_va, y_va in valid_loader:
                X_va, y_va = X_va.to(device), y_va.to(device)
                valid_batch_size = len(y_va)
                n_valid += valid_batch_size

                output_va = model(X_va)
                loss_va = criterion(output_va, y_va)

                pred_va = torch.argmax(output_va, dim=-1)
                running_valid_loss += loss_va.item() * valid_batch_size
                running_valid_acc += (pred_va == y_va).sum().item()

        train_losses = running_train_loss / n_train
        train_accuracies = running_train_acc / n_train
        valid_losses = running_valid_loss / n_valid
        valid_accuracies = running_valid_acc / n_valid

        item = np.array([int(epoch+1), train_losses, train_accuracies, valid_losses, valid_accuracies])
        history = np.vstack((history, item))
        print(f"[{epoch+1}/{num_epochs}] tr_loss: {history[-1, 1]:.5f}, tr_acc: {history[-1, 2]:.5f}, va_loss: {history[-1, 3]:.5f}, va_acc: {history[-1, 4]:.5f}")

        if valid_losses < best_loss:
            best_loss = valid_losses
            best_params = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
        
        if early_stopping and no_improve >= early_stopping:
            print(f"EARLY STOPPING -> epoch: {epoch+1}")
            break

    return history, best_params


def show_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(history[:, 0], history[:, 1], label='train')
    ax[0].plot(history[:, 0], history[:, 3], label='valid')
    ax[0].legend()
    ax[1].plot(history[:, 0], history[:, 2], label='train')
    ax[1].plot(history[:, 0], history[:, 4], label='valid')
    ax[1].legend()
    fig.tight_layout()
