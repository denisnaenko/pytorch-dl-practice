import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import ensure_results_dirs, get_classes, get_images

# Пути
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "train")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "analysis")
ensure_results_dirs()

# Сбор информации
class_counts = {}
img_sizes = []
size_by_class = defaultdict(list)

classes = get_classes(DATA_DIR)

for cls in classes:
    class_dir = os.path.join(DATA_DIR, cls)
    images = get_images(class_dir)
    class_counts[cls] = len(images)

    for img_name in images:
        img_path = os.path.join(class_dir, img_name)
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                img_sizes.append((w, h))
                size_by_class[cls].append((w, h))
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

# Анализ размеров
widths = [w for w, _ in img_sizes]
heights = [h for _, h in img_sizes]
areas = [w * h for w, h in img_sizes]

min_size = min(img_sizes, key=lambda x: x[0] * x[1])
max_size = max(img_sizes, key=lambda x: x[0] * x[1])
mean_w = np.mean(widths)
mean_h = np.mean(heights)
mean_area = np.mean(areas)

# Сохраняем статистику
with open(os.path.join(RESULTS_DIR, "dataset_stats.txt"), "w", encoding="utf-8") as f:
    f.write("Количество изображений по классам:\n")

    for cls, count in class_counts.items():
        f.write(f"{cls}: {count}\n")

    f.write(f"\nМинимальный размер: {min_size}\n")
    f.write(f"Максимальный размер: {max_size}\n")
    f.write(f"Средний размер: ({mean_w:.1f}, {mean_h:.1f})\n")
    f.write(f"Средняя площадь: {mean_area:.1f}\n")

# Визуализация: гистограмма по классам
plt.figure(figsize=(8, 4))
plt.bar(list(class_counts.keys()), list(class_counts.values()))
plt.title("Количество изображений по классам")
plt.ylabel("Количество")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "class_hist.png"))
plt.close()

# Визуализация: распределение размеров
plt.figure(figsize=(8, 4))
plt.hist(widths, bins=20, alpha=0.5, label="Ширина")
plt.hist(heights, bins=20, alpha=0.5, label="Высота")
plt.title("Распределение размеров изображений")
plt.xlabel("Пиксели")
plt.ylabel("Количество")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "size_hist.png"))
plt.close()

# Визуализация: scatter plot ширина/высота
plt.figure(figsize=(6, 6))
plt.scatter(widths, heights, alpha=0.3)
plt.xlabel("Ширина")
plt.ylabel("Высота")
plt.title("Ширина vs Высота изображений")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "wh_scatter.png"))
plt.close()

print("Анализ датасета завершён. Результаты сохранены в results/analysis/")
