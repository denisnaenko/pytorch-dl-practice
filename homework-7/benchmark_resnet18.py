import os
import pickle
import time

import GPUtil
import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
import torch
import torch_tensorrt
import torchvision.models as models

os.makedirs("onnx", exist_ok=True)

for size in [224, 256, 384, 512]:
    model = models.resnet18(pretrained=True)
    model.eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        model,
        dummy,  # type: ignore
        f"onnx/resnet18_{size}.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Exported ONNX for {size}x{size} with dynamic batch size")


os.makedirs("trt", exist_ok=True)

for size in [224, 256, 384, 512]:
    model = models.resnet18(pretrained=True).cuda().eval()
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 3, size, size),
                opt_shape=(8, 3, size, size),
                max_shape=(32, 3, size, size),
                dtype=torch.float32,
            )
        ],
        enabled_precisions={torch.float},
        workspace_size=1 << 22,
    )

    with open(f"trt/resnet18_{size}_trt.pkl", "wb") as f:
        pickle.dump(trt_model, f)
    print(f"Saved TensorRT model for {size}x{size} as pickle (dynamic batch)")


def benchmark_pytorch(size, batch_size, device):
    model = models.resnet18(pretrained=True).to(device).eval()
    x = torch.randn(batch_size, 3, size, size, device=device)
    torch.cuda.reset_peak_memory_stats()

    process = psutil.Process()
    cpu_mem_before = process.memory_info().rss

    # Прогрев
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        start = time.time()
        _ = model(x)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)
    cpu_mem = (process.memory_info().rss - cpu_mem_before) / (1024**2)

    return np.mean(times), np.std(times), gpu_mem, cpu_mem


def benchmark_onnx(size, batch_size):
    ort_sess = ort.InferenceSession(
        f"onnx/resnet18_{size}.onnx", providers=["CUDAExecutionProvider"]
    )
    x = np.random.randn(batch_size, 3, size, size).astype(np.float32)

    process = psutil.Process()
    cpu_mem_before = process.memory_info().rss

    # Прогрев
    for _ in range(10):
        _ = ort_sess.run(None, {"input": x})
    times = []
    for _ in range(50):
        start = time.time()
        _ = ort_sess.run(None, {"input": x})
        times.append(time.time() - start)

    # GPU memory через GPUtil
    gpus = GPUtil.getGPUs()
    gpu_mem = gpus[0].memoryUsed if gpus else np.nan
    cpu_mem = (process.memory_info().rss - cpu_mem_before) / (1024**2)

    return np.mean(times), np.std(times), gpu_mem, cpu_mem


def benchmark_trt(size, batch_size):
    with open(f"trt/resnet18_{size}_trt.pkl", "rb") as f:
        trt_model = pickle.load(f)

    x = torch.randn(batch_size, 3, size, size, device="cuda")
    torch.cuda.reset_peak_memory_stats()

    process = psutil.Process()
    cpu_mem_before = process.memory_info().rss

    # Прогрев
    for _ in range(10):
        _ = trt_model(x)
    torch.cuda.synchronize()
    times = []

    for _ in range(50):
        start = time.time()
        _ = trt_model(x)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    gpu_mem = torch.cuda.max_memory_allocated() / (1024**2)
    cpu_mem = (process.memory_info().rss - cpu_mem_before) / (1024**2)

    return np.mean(times), np.std(times), gpu_mem, cpu_mem


if __name__ == "__main__":
    device = "cuda"
    batch_sizes_dict = {
        224: [1, 2, 4, 8, 16, 32],
        256: [1, 2, 4, 8, 16, 32],
        384: [1, 2, 4, 8, 16],
        512: [1, 2, 4, 8],
    }
    sizes = [224, 256, 384, 512]
    results = []

    for size in sizes:
        for batch in batch_sizes_dict[size]:
            pt_mean, pt_std, pt_gpu_mem, pt_cpu_mem = benchmark_pytorch(
                size, batch, device
            )
            onnx_mean, onnx_std, onnx_gpu_mem, onnx_cpu_mem = benchmark_onnx(
                size, batch
            )
            try:
                trt_mean, trt_std, trt_gpu_mem, trt_cpu_mem = benchmark_trt(size, batch)
            except RuntimeError as e:
                print(f"OOM for size {size}, batch {batch}: {e}")
                trt_mean, trt_std, trt_gpu_mem, trt_cpu_mem = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            pt_fps = batch / pt_mean
            onnx_fps = batch / onnx_mean
            trt_fps = batch / trt_mean if not np.isnan(trt_mean) else np.nan
            results.append(
                {
                    "size": size,
                    "batch": batch,
                    "pt_mean": pt_mean,
                    "pt_std": pt_std,
                    "pt_fps": pt_fps,
                    "pt_gpu_mem": pt_gpu_mem,
                    "pt_cpu_mem": pt_cpu_mem,
                    "onnx_mean": onnx_mean,
                    "onnx_std": onnx_std,
                    "onnx_fps": onnx_fps,
                    "onnx_gpu_mem": onnx_gpu_mem,
                    "onnx_cpu_mem": onnx_cpu_mem,
                    "trt_mean": trt_mean,
                    "trt_std": trt_std,
                    "trt_fps": trt_fps,
                    "trt_gpu_mem": trt_gpu_mem,
                    "trt_cpu_mem": trt_cpu_mem,
                }
            )
            print(
                f"Size {size}, Batch {batch}: PT {pt_mean:.4f}s, ONNX {onnx_mean:.4f}s, TRT {trt_mean:.4f}s | PT FPS: {pt_fps:.2f}, ONNX FPS: {onnx_fps:.2f}, TRT FPS: {trt_fps if not np.isnan(trt_fps) else 'OOM'} | PT GPU Mem: {pt_gpu_mem:.1f}MB, ONNX GPU Mem: {onnx_gpu_mem:.1f}MB, TRT GPU Mem: {trt_gpu_mem if not np.isnan(trt_gpu_mem) else 'OOM'}MB"
            )

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("Results saved to benchmark_results.csv")
