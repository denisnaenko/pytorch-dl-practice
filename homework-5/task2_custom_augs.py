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
selected_classes = classes[:5]

# Для каждого класса выбираем одно изображение
sample_images = get_sample_images(DATA_DIR, n_classes=5, n_per_class=1)

# Кастомные аугментации
custom_augs = {
    "GaussianBlur": T.GaussianBlur(kernel_size=9, sigma=(0.1, 5)),
    "RandomPerspective": T.RandomPerspective(distortion_scale=0.5, p=1.0),
    "RandomBrightnessContrast": T.ColorJitter(brightness=0.8, contrast=0.8),
}

# Все кастомные аугментации вместе
all_custom_augs = T.Compose(
    [
        T.GaussianBlur(kernel_size=9, sigma=(0.1, 5)),
        T.RandomPerspective(distortion_scale=0.5, p=1.0),
        T.ColorJitter(brightness=0.8, contrast=0.8),
    ]
)

for cls, img_path in sample_images:
    orig = (
        Image.open(img_path).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    )
    aug_imgs = [orig]
    aug_titles = ["Original"]

    for name, aug in custom_augs.items():
        aug_img = aug(orig)
        aug_imgs.append(aug_img)
        aug_titles.append(name)

    all_aug_img = all_custom_augs(orig)
    aug_imgs.append(all_aug_img)
    aug_titles.append("All Custom Augs")

    save_path = os.path.join(RESULTS_DIR, f"custom_{cls}_augs.png")
    plot_and_save(aug_imgs, aug_titles, save_path)
    print(f"Saved: {save_path}")
