import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import json

# Hiperparametreler
batch_size = 64
epochs = 5
lr = 0.001
subset_size = 1000

# Veriyi yükle
transform = transforms.ToTensor()
mnist_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_subset, _ = random_split(mnist_full, [subset_size, len(mnist_full) - subset_size])
train_size = int(0.8 * subset_size)
val_size = subset_size - train_size
train_set, val_set = random_split(mnist_subset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# MLP Modeli
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Eğitim
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss / len(train_loader))

    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            epoch_val_loss += loss.item()
    val_losses.append(epoch_val_loss / len(val_loader))

# Grafiği kaydet
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("mlp_loss.png")

# Doğruluk raporu
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in val_loader:
        X = X.to(device)
        preds = model(X).argmax(1).cpu()
        all_preds.extend(preds)
        all_labels.extend(y)

report = classification_report(all_labels, all_preds, output_dict=True)
with open("mlp_metrics.json", "w") as f:
    json.dump(report, f, indent=2)
