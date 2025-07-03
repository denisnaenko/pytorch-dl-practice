import matplotlib.pyplot as plt
import seaborn as sns


def plot_learning_curves(history, save_path=None, title=None):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if title:
        plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_loss_curves(history, save_path=None, title=None):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if title:
        plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_heatmap(data, xlabels, ylabels, save_path=None, title=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt='.3f', xticklabels=xlabels,
                yticklabels=ylabels, cmap='viridis')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_weight_distribution(weights, save_path=None, title=None):
    plt.figure(figsize=(8, 5))
    sns.histplot(weights, bins=50, kde=True)
    plt.xlabel('Weight value')
    plt.ylabel('Frequency')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()
