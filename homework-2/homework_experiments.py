import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import DataLoader, TensorDataset
from typing import List
from homework_model_modification import LinearRegressionModified, LogisticRegressionModified, train_with_early_stopping, train_logistic_regression
from homework_datasets import create_diabetes_dataset, create_breast_cancer_dataset

# Создаем папку для графиков
os.makedirs('plots', exist_ok=True)


def run_lr_batchsize_experiment(model_class, dataset_func, task: str, lrs: List[float], batch_sizes: List[int], optimizer_names: List[str], epochs: int = 100):
    """
    Запускает эксперименты с различными скоростями обучения, размерами батчей и оптимизаторами.
    Возвращает DataFrame с результатами для компактного построения графиков.
    """
    results = []
    dataset = dataset_func()
    X_train, X_test, y_train, y_test = dataset.get_data()
    for lr in lrs:
        for batch_size in batch_sizes:
            for opt_name in optimizer_names:
                # Подготавливаем dataloader
                train_loader = DataLoader(TensorDataset(
                    X_train, y_train), batch_size=batch_size, shuffle=True)
                # Модель
                if task == 'regression':
                    model = model_class(
                        X_train.shape[1], l1_lambda=0.01, l2_lambda=0.1)
                else:
                    model = model_class(
                        X_train.shape[1], num_classes=2, l1_lambda=0.01, l2_lambda=0.1)
                # Оптимизатор
                if opt_name == 'SGD':
                    optimizer = torch.optim.SGD([model.w, model.b], lr=lr)
                elif opt_name == 'Adam':
                    optimizer = torch.optim.Adam([model.w, model.b], lr=lr)
                elif opt_name == 'RMSprop':
                    optimizer = torch.optim.RMSprop([model.w, model.b], lr=lr)
                else:
                    raise ValueError(f"Неизвестный оптимизатор: {opt_name}")
                # Цикл обучения
                for _ in range(epochs):
                    model.zero_grad()
                    for xb, yb in train_loader:
                        y_pred = model(xb)
                        _ = model.compute_loss(xb, yb, y_pred)
                        model.backward(xb, yb, y_pred)
                        optimizer.step()
                # Валидация
                with torch.no_grad():
                    y_pred_val = model(X_test)
                    val_loss = model.compute_loss(X_test, y_test, y_pred_val)
                results.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'optimizer': opt_name,
                    'final_val_loss': val_loss.item()
                })
    df = pd.DataFrame(results)
    # Строим групповой график: X=lr, hue=optimizer, group=batch_size
    plt.figure(figsize=(10, 6))
    for opt in optimizer_names:
        for batch in batch_sizes:
            subset = df[(df['optimizer'] == opt) & (df['batch_size'] == batch)]
            plt.plot(subset['lr'], subset['final_val_loss'],
                     marker='o', label=f'{opt}, batch={batch}')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Validation Loss')
    plt.title(f'Hyperparameter Search ({task.title()})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'plots/hyperparams_{task}.png')
    plt.close()
    return df


def add_polynomial_features(X: torch.Tensor, degree: int = 2) -> torch.Tensor:
    """Добавляет полиномиальные признаки к тензору X используя sklearn PolynomialFeatures."""
    poly = PolynomialFeatures(degree, include_bias=False)
    X_poly = poly.fit_transform(X.cpu().numpy())
    return torch.tensor(X_poly, dtype=torch.float32)


def add_interaction_features(X: torch.Tensor) -> torch.Tensor:
    """Добавляет попарные признаки взаимодействия (x_i * x_j для i < j)."""
    X_np = X.cpu().numpy()
    n = X_np.shape[1]
    features = [X_np]
    for i in range(n):
        for j in range(i+1, n):
            features.append((X_np[:, i] * X_np[:, j]).reshape(-1, 1))
    X_new = np.concatenate(features, axis=1)
    return torch.tensor(X_new, dtype=torch.float32)


def add_statistical_features(X: torch.Tensor) -> torch.Tensor:
    """Добавляет среднее значение и дисперсию как признаки."""
    X_np = X.cpu().numpy()
    mean_feat = X_np.mean(axis=1, keepdims=True)
    var_feat = X_np.var(axis=1, keepdims=True)
    X_new = np.concatenate([X_np, mean_feat, var_feat], axis=1)
    return torch.tensor(X_new, dtype=torch.float32)


def compare_feature_engineering(dataset_func, model_class, task: str):
    """
    Сравнивает базовую модель и модели с инженерными признаками. Строит barplot.
    """
    dataset = dataset_func()
    X_train, X_test, y_train, y_test = dataset.get_data()
    results = {}

    # Базовая модель
    model = model_class(
        X_train.shape[1], num_classes=2) if task == 'classification' else model_class(X_train.shape[1])
    history = train_with_early_stopping(model, X_train, y_train, X_test, y_test, epochs=200, lr=0.01, verbose=False) if task == 'regression' else train_logistic_regression(
        model, X_train, y_train, X_test, y_test, epochs=200, lr=0.1, verbose=False)
    results['base'] = history['val_losses'][-1]

    # Полиномиальные признаки
    X_train_poly = add_polynomial_features(X_train, degree=2)
    X_test_poly = add_polynomial_features(X_test, degree=2)
    model_poly = model_class(
        X_train_poly.shape[1], num_classes=2) if task == 'classification' else model_class(X_train_poly.shape[1])
    history_poly = train_with_early_stopping(model_poly, X_train_poly, y_train, X_test_poly, y_test, epochs=200, lr=0.01, verbose=False) if task == 'regression' else train_logistic_regression(
        model_poly, X_train_poly, y_train, X_test_poly, y_test, epochs=200, lr=0.1, verbose=False)
    results['poly'] = history_poly['val_losses'][-1]

    # Признаки взаимодействия
    X_train_inter = add_interaction_features(X_train)
    X_test_inter = add_interaction_features(X_test)
    model_inter = model_class(
        X_train_inter.shape[1], num_classes=2) if task == 'classification' else model_class(X_train_inter.shape[1])
    history_inter = train_with_early_stopping(model_inter, X_train_inter, y_train, X_test_inter, y_test, epochs=200, lr=0.01, verbose=False) if task == 'regression' else train_logistic_regression(
        model_inter, X_train_inter, y_train, X_test_inter, y_test, epochs=200, lr=0.1, verbose=False)
    results['interaction'] = history_inter['val_losses'][-1]

    # Статистические признаки
    X_train_stat = add_statistical_features(X_train)
    X_test_stat = add_statistical_features(X_test)
    model_stat = model_class(
        X_train_stat.shape[1], num_classes=2) if task == 'classification' else model_class(X_train_stat.shape[1])
    history_stat = train_with_early_stopping(model_stat, X_train_stat, y_train, X_test_stat, y_test, epochs=200, lr=0.01, verbose=False) if task == 'regression' else train_logistic_regression(
        model_stat, X_train_stat, y_train, X_test_stat, y_test, epochs=200, lr=0.1, verbose=False)
    results['stat'] = history_stat['val_losses'][-1]

    # Plot concise barplot
    plt.figure(figsize=(7, 5))
    keys = list(results.keys())
    vals = [results[k] for k in keys]
    plt.bar(keys, vals, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    plt.title(f'Feature Engineering Comparison ({task})')
    plt.ylabel('Final Validation Loss')
    plt.xlabel('Feature Set')
    plt.tight_layout()
    plt.savefig(f'plots/feature_engineering_{task}.png')
    plt.close()
    return results


# Пример использования
if __name__ == "__main__":
    # 1. Эксперименты с гиперпараметрами (классификация)
    run_lr_batchsize_experiment(
        LogisticRegressionModified,
        create_breast_cancer_dataset,
        task='classification',
        lrs=[0.01, 0.05, 0.1],
        batch_sizes=[16, 32, 64],
        optimizer_names=['SGD', 'Adam', 'RMSprop'],
        epochs=50
    )
    # 2. Feature engineering (classification)
    compare_feature_engineering(
        create_breast_cancer_dataset, LogisticRegressionModified, task='classification')
    # 3. Эксперименты с гиперпараметрами (регрессия)
    run_lr_batchsize_experiment(
        LinearRegressionModified,
        create_diabetes_dataset,
        task='regression',
        lrs=[0.001, 0.01, 0.05],
        batch_sizes=[16, 32, 64],
        optimizer_names=['SGD', 'Adam', 'RMSprop'],
        epochs=50
    )
    # 4. Feature engineering (regression)
    compare_feature_engineering(
        create_diabetes_dataset, LinearRegressionModified, task='regression')
