import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from datasets import CustomImageDataset

# Пути
TRAIN_DIR = os.path.join(os.path.dirname(__file__), "data", "train")
VAL_DIR = os.path.join(os.path.dirname(__file__), "data", "val")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "finetune")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Трансформации
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Датасеты и загрузчики
train_dataset = CustomImageDataset(TRAIN_DIR, transform=transform)
val_dataset = CustomImageDataset(VAL_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.get_class_names()))
model = model.to(device)

# Оптимизатор и функция потерь
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Обучение
num_epochs = 5
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        _, preds = torch.max(out, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Валидация
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            running_loss += loss.item() * x.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch+1}/{num_epochs} | Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
    )

# Сохраняем модель
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "finetuned_resnet18.pth"))

# Визуализация процесса обучения
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label="Train loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss during training")
plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
plt.close()

plt.figure()
plt.plot(range(1, num_epochs + 1), train_accs, label="Train acc")
plt.plot(range(1, num_epochs + 1), val_accs, label="Val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy during training")
plt.savefig(os.path.join(RESULTS_DIR, "acc_curve.png"))
plt.close()

print("Finetuning завершён. Модель и графики сохранены в results/finetune/")
