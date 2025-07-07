import os

import torchvision.transforms as T
from PIL import Image

from utils import ensure_results_dirs, get_sample_images, plot_and_save

# Пути
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "train")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "augs")
ensure_results_dirs()

# Получаем список классов
classes = sorted(
    [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
)

# Выбираем 5 разных классов
selected_classes = classes[:5]

# Для каждого класса выбираем одно изображение
sample_images = get_sample_images(DATA_DIR, n_classes=5, n_per_class=1)

# Стандартные аугментации torchvision
augs = {
    "RandomHorizontalFlip": T.RandomHorizontalFlip(p=1.0),
    "RandomCrop": T.RandomCrop(224, padding=32),
    "ColorJitter": T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    "RandomRotation": T.RandomRotation(degrees=45),
    "RandomGrayscale": T.RandomGrayscale(p=1.0),
}

# Все аугментации вместе
all_augs = T.Compose(
    [
        T.RandomHorizontalFlip(p=1.0),
        T.RandomCrop(224, padding=32),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        T.RandomRotation(degrees=45),
        T.RandomGrayscale(p=1.0),
    ]
)

for cls, img_path in sample_images:
    orig = (
        Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    )
    aug_imgs = [orig]
    aug_titles = ["Original"]

    # Каждая аугментация отдельно
    for name, aug in augs.items():
        aug_img = aug(orig)
        aug_imgs.append(aug_img)
        aug_titles.append(name)

    # Все аугментации вместе
    all_aug_img = all_augs(orig)
    aug_imgs.append(all_aug_img)
    aug_titles.append("All Augs")

    # Сохраняем результат
    save_path = os.path.join(RESULTS_DIR, f"{cls}_augs.png")
    plot_and_save(aug_imgs, aug_titles, save_path)

    print(f"Saved: {save_path}")
