import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from utils.experiment_utils import run_experiment
from utils.model_utils import count_parameters, create_mlp
from utils.visualization_utils import plot_learning_curves, plot_loss_curves

# Настройка логирования
logging.basicConfig(
    filename='results/depth_experiments/experiment.log', level=logging.INFO)

# Гиперпараметры
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = 'results/depth_experiments/'
PLOTS_DIR = 'plots/depth_experiments/'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Загрузка данных MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
train_dataset = datasets.MNIST(
    root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(
    root='data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

input_dim = 28*28
output_dim = 10

depths = {
    '1-layer': [],
    '2-layer': [128],
    '3-layer': [256, 128],
    '5-layer': [512, 256, 128, 64],
    '7-layer': [512, 256, 128, 64, 32, 16]
}

# 1.1 Сравнение моделей разной глубины
for name, hidden_layers in depths.items():
    model = create_mlp(input_dim, output_dim, hidden_layers).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    logging.info(f"Training {name} ({count_parameters(model)} params)")

    history = run_experiment(
        model, train_loader, test_loader, criterion, optimizer, DEVICE, epochs=EPOCHS)
    plot_learning_curves(
        history, save_path=f"{PLOTS_DIR}{name}_acc.png", title=f"{name} Accuracy")
    plot_loss_curves(
        history, save_path=f"{PLOTS_DIR}{name}_loss.png", title=f"{name} Loss")
    torch.save(model.state_dict(), f"{RESULTS_DIR}{name}_model.pth")

# 1.2 Анализ переобучения: Dropout и BatchNorm
for reg_type in ['dropout', 'batchnorm', 'dropout+batchnorm']:
    for name, hidden_layers in depths.items():
        use_dropout = reg_type in ['dropout', 'dropout+batchnorm']
        use_batchnorm = reg_type in ['batchnorm', 'dropout+batchnorm']

        model = create_mlp(input_dim, output_dim, hidden_layers, use_dropout=use_dropout,
                           dropout_p=0.3, use_batchnorm=use_batchnorm).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        logging.info(f"Training {name} with {reg_type}")

        history = run_experiment(
            model, train_loader, test_loader, criterion, optimizer, DEVICE, epochs=EPOCHS)
        plot_learning_curves(
            history, save_path=f"{PLOTS_DIR}{name}_{reg_type}_acc.png", title=f"{name} {reg_type} Accuracy")
        plot_loss_curves(
            history, save_path=f"{PLOTS_DIR}{name}_{reg_type}_loss.png", title=f"{name} {reg_type} Loss")

        torch.save(model.state_dict(),
                   f"{RESULTS_DIR}{name}_{reg_type}_model.pth")
