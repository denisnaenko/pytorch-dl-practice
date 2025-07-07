import os
import random

import matplotlib.pyplot as plt


def plot_and_save(imgs, titles, save_path, figsize=(15, 3)):
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, len(imgs), i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_classes(data_dir):
    """Возвращает отсортированный список классов (папок)"""
    return sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )


def get_images(class_dir):
    """Возвращает список файлов-изображений в папке класса"""
    return [
        f
        for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]


def get_sample_images(data_dir, n_classes=5, n_per_class=1, seed=42):
    """Возвращает список (class, img_path) для n_classes классов и n_per_class изображений в каждом"""
    random.seed(seed)

    classes = get_classes(data_dir)[:n_classes]
    sample_images = []

    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        images = get_images(class_dir)
        for img_name in random.sample(images, min(n_per_class, len(images))):
            img_path = os.path.join(class_dir, img_name)
            sample_images.append((cls, img_path))

    return sample_images


def ensure_results_dirs():
    """Создает структуру results/ и подпапок, если их нет"""
    base = os.path.join(os.path.dirname(__file__), "results")

    for sub in ["augs", "analysis", "pipeline", "resize_exp", "finetune"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
