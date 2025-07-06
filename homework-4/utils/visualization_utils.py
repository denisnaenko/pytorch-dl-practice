import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def plot_training_history(history, title="Training History", save_path=None):
    """Визуализирует историю обучения"""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history["train_losses"]) + 1)

    # График потерь
    ax1.plot(epochs, history["train_losses"],
             "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["test_losses"],
             "r-", label="Test Loss", linewidth=2)
    ax1.set_title("Loss History", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График точности
    ax2.plot(epochs, history["train_accs"], "b-",
             label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, history["test_accs"], "r-",
             label="Test Accuracy", linewidth=2)
    ax2.set_title("Accuracy History", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_model_comparison(
    results_dict, metric="test_accs", title="Model Comparison", save_path=None
):
    """Сравнивает несколько моделей по заданной метрике"""
    _, ax = plt.subplots(figsize=(12, 6))

    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    markers = ["o", "s", "^", "D", "v", "<"]

    for i, (model_name, history) in enumerate(results_dict.items()):
        epochs = range(1, len(history[metric]) + 1)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.plot(
            epochs,
            history[metric],
            color=color,
            marker=marker,
            label=model_name,
            linewidth=2,
            markersize=6,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    predictions, targets, class_names=None, title="Confusion Matrix", save_path=None
):
    """Визуализирует матрицу ошибок"""
    cm = confusion_matrix(targets, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,  # type: ignore
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_parameter_comparison(models_info, save_path=None):
    """Сравнивает количество параметров моделей"""
    model_names = list(models_info.keys())
    param_counts = [info["total_params_millions"]
                    for info in models_info.values()]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        model_names, param_counts, color="skyblue", edgecolor="navy", alpha=0.7
    )

    # Добавляем значения на столбцы
    for bar, count in zip(bars, param_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{count:.2f}M",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.title("Model Parameters Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel("Parameters (Millions)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis="y")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_inference_time_comparison(models_info, save_path=None):
    """Сравнивает время инференса моделей"""
    model_names = list(models_info.keys())
    mean_times = [info["inference_time"]["mean_time"]
                  for info in models_info.values()]

    plt.figure(figsize=(max(8, len(model_names) * 2), 6))
    bars = plt.bar(
        model_names, mean_times, color="lightcoral", edgecolor="darkred", alpha=0.7
    )

    # Добавляем значения на столбцы
    for bar, time_val in zip(bars, mean_times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(mean_times),
            f"{time_val:.4f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            rotation=0,
        )

    plt.title("Model Inference Time Comparison",
              fontsize=14, fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel("Inference Time (seconds)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis="y")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_feature_maps(
    model, data_loader, layer_name, num_images=4, num_channels=8, save_path=None
):
    """Визуализирует feature maps первого слоя красиво и компактно"""
    model.eval()
    for data, _ in data_loader:
        break
    device = next(model.parameters()).device
    data = data.to(device)

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(get_activation(layer_name))
            break

    with torch.no_grad():
        _ = model(data[:num_images])

    if layer_name not in activations:
        print(f"Layer {layer_name} not found!")
        return

    feature_maps = activations[layer_name].cpu()
    n_img = min(num_images, feature_maps.shape[0])
    n_ch = min(num_channels, feature_maps.shape[1])

    fig, axes = plt.subplots(n_img, n_ch, figsize=(2 * n_ch, 2 * n_img))
    if n_img == 1:
        axes = axes[np.newaxis, :]
    if n_ch == 1:
        axes = axes[:, np.newaxis]
    for i in range(n_img):
        for j in range(n_ch):
            ax = axes[i, j]
            ax.imshow(feature_maps[i, j], cmap="viridis")
            ax.axis("off")
    plt.suptitle(f"Feature Maps from {layer_name}",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_gradient_flow(model, data_loader, save_path=None):
    """Визуализирует поток градиентов"""
    model.train()

    # Получаем данные
    for data, target in data_loader:
        break

    device = next(model.parameters()).device
    data = data.to(device)
    target = target.to(device)

    # Обучаем модель на одном батче
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

    # Собираем градиенты
    gradients = []
    layer_names = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients.append(param.grad.abs().mean().item())
            layer_names.append(name)

    # Визуализируем
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(gradients)), gradients, color="orange", alpha=0.7)
    plt.title("Gradient Flow Analysis", fontsize=14, fontweight="bold")
    plt.xlabel("Layer")
    plt.ylabel("Average Gradient Magnitude")
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.yscale("log")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_summary_table(results_dict, save_path=None):
    """Создает сводную таблицу результатов"""
    import pandas as pd

    summary_data = []

    for model_name, results in results_dict.items():
        summary_data.append(
            {
                "Model": model_name,
                "Final Train Acc": f"{results['train_accs'][-1]:.4f}",
                "Final Test Acc": f"{results['test_accs'][-1]:.4f}",
                "Best Test Acc": f"{max(results['test_accs']):.4f}",
                "Final Train Loss": f"{results['train_losses'][-1]:.4f}",
                "Final Test Loss": f"{results['test_losses'][-1]:.4f}",
                "Training Time": f"{results.get('total_time', 0):.2f}s",
                "Parameters": f"{results.get('parameters', {}).get('total_params_millions', 0):.2f}M",
            }
        )

    df = pd.DataFrame(summary_data)

    if save_path:
        df.to_csv(save_path, index=False)

    return df
