import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
    )


def run_epoch(
    model, data_loader, criterion, optimizer=None, device="cpu", is_test=False
):
    """Выполняет одну эпоху обучения или тестирования"""
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0
    predictions = []
    targets = []

    with torch.no_grad() if is_test else torch.enable_grad():
        for batch_idx, (data, target) in enumerate(
            tqdm(data_loader, desc=f"{'Test' if is_test else 'Train'}")
        ):
            data, target = data.to(device), target.to(device)

            if not is_test and optimizer is not None:
                optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            if not is_test and optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            if is_test:
                predictions.extend(pred.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)

    return avg_loss, accuracy, predictions, targets


def train_model(
    model,
    train_loader,
    test_loader,
    epochs=10,
    lr=0.001,
    device="cpu",
    optimizer_name="adam",
    weight_decay=0.0,
):
    """Обучает модель и возвращает историю обучения"""
    setup_logging()

    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, is_test=False
        )
        test_loss, test_acc, predictions, targets = run_epoch(
            model, test_loader, criterion, None, device, is_test=True
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        epoch_time = time.time() - epoch_start

        logging.info(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s):")
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logging.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        logging.info("-" * 50)

    total_time = time.time() - start_time

    return {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
        "predictions": predictions,
        "targets": targets,
        "total_time": total_time,
    }


def measure_inference_time(model, data_loader, device="cpu", num_runs=100):
    """Измеряет время инференса модели"""
    model.eval()
    model.to(device)

    # Прогрев
    for data, _ in data_loader:
        data = data.to(device)
        with torch.no_grad():
            _ = model(data)
        break

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            for data, _ in data_loader:
                data = data.to(device)
                start_time = time.time()
                _ = model(data)
                end_time = time.time()
                times.append(end_time - start_time)
                break

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
    }


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_millions": total_params / 1e6,
    }


def save_training_results(results, filename):
    """Сохраняет результаты обучения"""
    import pickle

    with open(filename, "wb") as f:
        pickle.dump(results, f)


def load_training_results(filename):
    """Загружает результаты обучения"""
    import pickle

    with open(filename, "rb") as f:
        return pickle.load(f)
