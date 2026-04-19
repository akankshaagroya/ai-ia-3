import os
import torch
import torch.nn as nn


class SymbolCNN(nn.Module):
    """3-layer CNN for 15-class symbol classification (28x28 grayscale)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 15)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = torch.relu(x)
        x = self.drop_fc(x)
        x = self.fc2(x)

        return x


def build_model():
    """Build and return a new model."""
    return SymbolCNN()


def save_model(model, path='models/equation_classifier.pt'):
    """Save trained model to disk."""
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path='models/equation_classifier.pt', device='cpu'):
    """Load trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
    model = SymbolCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
