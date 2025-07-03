import logging
import time
from typing import List

import torch
import torch.nn as nn


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)

        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def run_experiment(model, train_loader, test_loader, criterion, optimizer, device, epochs=20, log_interval=1):
    history = {'train_loss': [], 'train_acc': [],
               'test_loss': [], 'test_acc': [], 'epoch_time': []}
    for epoch in range(epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(elapsed)

        if epoch % log_interval == 0:
            logging.info(
                f"Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, Time={elapsed:.2f}s")

    return history


def get_weight_distribution(model: nn.Module) -> List[float]:
    weights = []

    for p in model.parameters():
        if p.requires_grad and p.dim() > 1:
            weights.extend(p.detach().cpu().numpy().flatten())

    return weights
