import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from utils.experiment_utils import run_experiment
from utils.model_utils import count_parameters, create_mlp
from utils.visualization_utils import (plot_heatmap, plot_learning_curves,
                                       plot_loss_curves)

logging.basicConfig(
    filename='results/width_experiments/experiment.log', level=logging.INFO)

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = 'results/width_experiments/'
PLOTS_DIR = 'plots/width_experiments/'
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

width_configs = {
    'narrow': [64, 32, 16],
    'medium': [256, 128, 64],
    'wide': [1024, 512, 256],
    'very_wide': [2048, 1024, 512]
}

# 2.1 Сравнение моделей разной ширины
for name, hidden_layers in width_configs.items():
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
    with open(f"{RESULTS_DIR}{name}_params.txt", 'w') as f:
        f.write(f"Params: {count_parameters(model)}\n")
        f.write(f"Final test accuracy: {history['test_acc'][-1]:.4f}\n")

# 2.2 Grid search по ширине
width_grid = [32, 64, 128, 256, 512, 1024]
results = np.zeros((len(width_grid), len(width_grid)))
for i, w1 in enumerate(width_grid):
    for j, w2 in enumerate(width_grid):
        hidden_layers = [w1, w2, w1]
        model = create_mlp(input_dim, output_dim, hidden_layers).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        history = run_experiment(
            model, train_loader, test_loader, criterion, optimizer, DEVICE, epochs=10)
        acc = history['test_acc'][-1]
        results[i, j] = acc

        logging.info(f"Grid {w1}-{w2}-{w1}: acc={acc:.4f}")
plot_heatmap(results, xlabels=width_grid, ylabels=width_grid,
             save_path=f"{PLOTS_DIR}width_grid_heatmap.png", title="Grid Search Test Accuracy")
