import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)
df = pd.read_csv("benchmark_results.csv")
df["onnx_speedup"] = df["onnx_fps"] / df["pt_fps"]
df["trt_speedup"] = df["trt_fps"] / df["pt_fps"]

# 1. FPS vs Размер изображения
plt.figure(figsize=(8, 6))
for col, label in zip(
    ["pt_fps", "onnx_fps", "trt_fps"], ["PyTorch", "ONNX", "TensorRT"]
):
    sns.lineplot(x="size", y=col, data=df, marker="o", label=label)
plt.xlabel("Размер изображения")
plt.ylabel("FPS")
plt.title("FPS vs Размер изображения")
plt.legend()
plt.savefig("plots/fps_vs_size.png")
plt.close()

# 2. FPS vs Размер батча
plt.figure(figsize=(8, 6))
for col, label in zip(
    ["pt_fps", "onnx_fps", "trt_fps"], ["PyTorch", "ONNX", "TensorRT"]
):
    sns.lineplot(x="batch", y=col, data=df, marker="o", label=label)
plt.xlabel("Размер батча")
plt.ylabel("FPS")
plt.title("FPS vs Размер батча")
plt.legend()
plt.savefig("plots/fps_vs_batch.png")
plt.close()

# 3. Ускорение относительно PyTorch
plt.figure(figsize=(8, 6))
for col, label in zip(["onnx_speedup", "trt_speedup"], ["ONNX", "TensorRT"]):
    sns.lineplot(x="size", y=col, data=df, marker="o", label=label)
plt.xlabel("Размер изображения")
plt.ylabel("Ускорение относительно PyTorch")
plt.title("Ускорение (FPS) vs Размер изображения")
plt.legend()
plt.savefig("plots/speedup_vs_size.png")
plt.close()

plt.figure(figsize=(8, 6))
for col, label in zip(["onnx_speedup", "trt_speedup"], ["ONNX", "TensorRT"]):
    sns.lineplot(x="batch", y=col, data=df, marker="o", label=label)
plt.xlabel("Размер батча")
plt.ylabel("Ускорение относительно PyTorch")
plt.title("Ускорение (FPS) vs Размер батча")
plt.legend()
plt.savefig("plots/speedup_vs_batch.png")
plt.close()

# 4. GPU память vs Размер изображения
plt.figure(figsize=(8, 6))
for col, label in zip(
    ["pt_gpu_mem", "onnx_gpu_mem", "trt_gpu_mem"], ["PyTorch", "ONNX", "TensorRT"]
):
    sns.lineplot(x="size", y=col, data=df, marker="o", label=label)
plt.xlabel("Размер изображения")
plt.ylabel("GPU память (MB)")
plt.title("GPU память vs Размер изображения")
plt.legend()
plt.savefig("plots/gpu_mem_vs_size.png")
plt.close()

# 5. GPU память vs Размер батча
plt.figure(figsize=(8, 6))
for col, label in zip(
    ["pt_gpu_mem", "onnx_gpu_mem", "trt_gpu_mem"], ["PyTorch", "ONNX", "TensorRT"]
):
    sns.lineplot(x="batch", y=col, data=df, marker="o", label=label)
plt.xlabel("Размер батча")
plt.ylabel("GPU память (MB)")
plt.title("GPU память vs Размер батча")
plt.legend()
plt.savefig("plots/gpu_mem_vs_batch.png")
plt.close()

# 6. CPU память vs Размер изображения
plt.figure(figsize=(8, 6))
for col, label in zip(
    ["pt_cpu_mem", "onnx_cpu_mem", "trt_cpu_mem"], ["PyTorch", "ONNX", "TensorRT"]
):
    sns.lineplot(x="size", y=col, data=df, marker="o", label=label)
plt.xlabel("Размер изображения")
plt.ylabel("CPU память (MB)")
plt.title("CPU память vs Размер изображения")
plt.legend()
plt.savefig("plots/cpu_mem_vs_size.png")
plt.close()

# 7. CPU память vs Размер батча
plt.figure(figsize=(8, 6))
for col, label in zip(
    ["pt_cpu_mem", "onnx_cpu_mem", "trt_cpu_mem"], ["PyTorch", "ONNX", "TensorRT"]
):
    sns.lineplot(x="batch", y=col, data=df, marker="o", label=label)
plt.xlabel("Размер батча")
plt.ylabel("CPU память (MB)")
plt.title("CPU память vs Размер батча")
plt.legend()
plt.savefig("plots/cpu_mem_vs_batch.png")
plt.close()

print("Plots saved to plots/")
