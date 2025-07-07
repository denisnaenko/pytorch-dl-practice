import os
import random
import time

import matplotlib.pyplot as plt
import psutil
from PIL import Image
from torchvision import transforms

from utils import ensure_results_dirs, get_classes, get_images

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "train")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "resize_exp")
ensure_results_dirs()

sizes = [64, 128, 224, 512]
aug = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
    ]
)

classes = get_classes(DATA_DIR)
all_images = []
for cls in classes:
    class_dir = os.path.join(DATA_DIR, cls)
    images = [os.path.join(class_dir, f) for f in get_images(class_dir)]
    all_images.extend(images)

sample_images = random.sample(all_images, min(100, len(all_images)))

results = []
for size in sizes:
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    start = time.time()

    for img_path in sample_images:
        img = Image.open(img_path).convert("RGB")
        transform = transforms.Compose([transforms.Resize((size, size)), aug])
        _ = transform(img)

    elapsed = time.time() - start
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_used = mem_after - mem_before
    results.append({"size": size, "time": elapsed, "mem": mem_used})
    print(f"Size: {size}, Time: {elapsed:.2f}s, Mem: {mem_used:.2f}MB")

sizes_ = [r["size"] for r in results]
times_ = [r["time"] for r in results]
mems_ = [r["mem"] for r in results]

plt.figure()
plt.plot(sizes_, times_, marker="o")
plt.title("Время обработки 100 изображений vs Размер")
plt.xlabel("Размер (px)")
plt.ylabel("Время (сек)")
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "time_vs_size.png"))
plt.close()

plt.figure()
plt.plot(sizes_, mems_, marker="o")
plt.title("Память vs Размер изображений")
plt.xlabel("Размер (px)")
plt.ylabel("Память (MB)")
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "mem_vs_size.png"))
plt.close()

with open(
    os.path.join(RESULTS_DIR, "resize_experiment.txt"), "w", encoding="utf-8"
) as f:
    for r in results:
        f.write(f"Size: {r['size']}, Time: {r['time']:.2f}s, Mem: {r['mem']:.2f}MB\n")

print("Эксперимент завершён. Результаты сохранены в results/resize_exp/")
