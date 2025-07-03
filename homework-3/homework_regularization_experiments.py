import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from utils.experiment_utils import get_weight_distribution, run_experiment
from utils.model_utils import create_mlp
from utils.visualization_utils import (plot_learning_curves, plot_loss_curves,
                                       plot_weight_distribution)

logging.basicConfig(
    filename='results/regularization_experiments/experiment.log', level=logging.INFO)

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = 'results/regularization_experiments/'
PLOTS_DIR = 'plots/regularization_experiments/'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

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
hidden_layers = [256, 128, 64]  # одинаковая архитектура для всех экспериментов

# 3.1 Сравнение техник регуляризации
reg_configs = [
    {'name': 'no_reg', 'dropout': False, 'dropout_p': 0.0,
        'batchnorm': False, 'weight_decay': 0.0},
    {'name': 'dropout_0.1', 'dropout': True, 'dropout_p': 0.1,
        'batchnorm': False, 'weight_decay': 0.0},
    {'name': 'dropout_0.3', 'dropout': True, 'dropout_p': 0.3,
        'batchnorm': False, 'weight_decay': 0.0},
    {'name': 'dropout_0.5', 'dropout': True, 'dropout_p': 0.5,
        'batchnorm': False, 'weight_decay': 0.0},
    {'name': 'batchnorm', 'dropout': False, 'dropout_p': 0.0,
        'batchnorm': True, 'weight_decay': 0.0},
    {'name': 'dropout+batchnorm', 'dropout': True,
        'dropout_p': 0.3, 'batchnorm': True, 'weight_decay': 0.0},
    {'name': 'l2', 'dropout': False, 'dropout_p': 0.0,
        'batchnorm': False, 'weight_decay': 1e-3},
]

for cfg in reg_configs:
    model = create_mlp(input_dim, output_dim, hidden_layers,
                       use_dropout=cfg['dropout'], dropout_p=cfg['dropout_p'], use_batchnorm=cfg['batchnorm']).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR,
                           weight_decay=cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    logging.info(f"Training {cfg['name']}")

    history = run_experiment(
        model, train_loader, test_loader, criterion, optimizer, DEVICE, epochs=EPOCHS)

    plot_learning_curves(
        history, save_path=f"{PLOTS_DIR}{cfg['name']}_acc.png", title=f"{cfg['name']} Accuracy")
    plot_loss_curves(
        history, save_path=f"{PLOTS_DIR}{cfg['name']}_loss.png", title=f"{cfg['name']} Loss")
    weights = get_weight_distribution(model)
    plot_weight_distribution(
        weights, save_path=f"{PLOTS_DIR}{cfg['name']}_weights.png", title=f"{cfg['name']} Weights Distribution")
    torch.save(model.state_dict(), f"{RESULTS_DIR}{cfg['name']}_model.pth")


# 3.2 Адаптивная регуляризация: Dropout с изменяющимся коэффициентом
class AdaptiveDropoutMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_start=0.5, dropout_end=0.1, use_batchnorm=False):
        super().__init__()

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_dim = input_dim
        n_layers = len(hidden_layers)

        for i, h in enumerate(hidden_layers):
            self.layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                self.layers.append(nn.BatchNorm1d(h))
            self.layers.append(nn.ReLU())
            p = dropout_start + (dropout_end - dropout_start) * \
                (i / max(1, n_layers-1))
            self.dropouts.append(nn.Dropout(p))
            prev_dim = h

        self.out = nn.Linear(prev_dim, output_dim)
        self.use_batchnorm = use_batchnorm

    def forward(self, x):
        layer_idx = 0
        for i in range(len(self.dropouts)):
            x = self.layers[layer_idx](x)  # Linear
            layer_idx += 1

            if self.use_batchnorm:
                x = self.layers[layer_idx](x)  # BatchNorm
                layer_idx += 1

            x = self.layers[layer_idx](x)  # ReLU
            layer_idx += 1

            x = self.dropouts[i](x)  # Dropout

        x = self.out(x)
        return x


# Пример адаптивного Dropout
model = AdaptiveDropoutMLP(input_dim, output_dim, hidden_layers,
                           dropout_start=0.5, dropout_end=0.1).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

history = run_experiment(model, train_loader, test_loader,
                         criterion, optimizer, DEVICE, epochs=EPOCHS)

plot_learning_curves(
    history, save_path=f"{PLOTS_DIR}adaptive_dropout_acc.png", title="Adaptive Dropout Accuracy")
plot_loss_curves(
    history, save_path=f"{PLOTS_DIR}adaptive_dropout_loss.png", title="Adaptive Dropout Loss")
weights = get_weight_distribution(model)
plot_weight_distribution(
    weights, save_path=f"{PLOTS_DIR}adaptive_dropout_weights.png", title="Adaptive Dropout Weights Distribution")
torch.save(model.state_dict(), f"{RESULTS_DIR}adaptive_dropout_model.pth")
