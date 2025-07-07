import os
import random

import torchvision.transforms as T
from PIL import Image

from utils import ensure_results_dirs, get_classes, get_images, plot_and_save


class AugmentationPipeline:
    def __init__(self):
        self.augmentations = []
        self.aug_names = []

    def add_augmentation(self, name, aug):
        self.aug_names.append(name)
        self.augmentations.append(aug)

    def remove_augmentation(self, name):
        if name in self.aug_names:
            idx = self.aug_names.index(name)
            self.aug_names.pop(idx)
            self.augmentations.pop(idx)

    def apply(self, image):
        img = image.copy()
        for aug in self.augmentations:
            img = aug(img)
        return img

    def get_augmentations(self):
        return list(zip(self.aug_names, self.augmentations))


# Конфигурации
configs = {
    "light": [
        ("RandomHorizontalFlip", T.RandomHorizontalFlip(p=1.0)),
    ],
    "medium": [
        ("RandomHorizontalFlip", T.RandomHorizontalFlip(p=1.0)),
        ("ColorJitter", T.ColorJitter(brightness=0.3, contrast=0.3)),
        ("RandomRotation", T.RandomRotation(degrees=20)),
    ],
    "heavy": [
        ("RandomHorizontalFlip", T.RandomHorizontalFlip(p=1.0)),
        (
            "ColorJitter",
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        ),
        ("RandomRotation", T.RandomRotation(degrees=45)),
        ("RandomPerspective", T.RandomPerspective(distortion_scale=0.5, p=1.0)),
        ("GaussianBlur", T.GaussianBlur(kernel_size=7, sigma=(0.1, 3))),
    ],
}

# Пути
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "train")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "pipeline")
ensure_results_dirs()

# Получаем список классов и по 2 изображения из каждого
classes = get_classes(DATA_DIR)
sample_images = []
for cls in classes:
    class_dir = os.path.join(DATA_DIR, cls)
    images = get_images(class_dir)
    for img_name in random.sample(images, min(2, len(images))):
        img_path = os.path.join(class_dir, img_name)
        sample_images.append((cls, img_path))

# Применяем пайплайны к изображениям
for config_name, aug_list in configs.items():
    pipeline = AugmentationPipeline()

    for name, aug in aug_list:
        pipeline.add_augmentation(name, aug)

    for cls, img_path in random.sample(sample_images, min(5, len(sample_images))):
        orig = (
            Image.open(img_path)
            .convert("RGB")
            .resize((224, 224), Image.Resampling.LANCZOS)
        )

        aug_img = pipeline.apply(orig)
        imgs = [orig, aug_img]
        titles = ["Original", f"{config_name} pipeline"]

        save_path = os.path.join(
            RESULTS_DIR, f"{config_name}_{cls}_{os.path.basename(img_path)}"
        )

        plot_and_save(imgs, titles, save_path)
        print(f"Saved: {save_path}")
