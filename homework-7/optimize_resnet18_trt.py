import os
import pickle

import torch
import torch_tensorrt
import torchvision.models as models

os.makedirs("trt", exist_ok=True)

for size in [224, 256, 384, 512]:
    model = models.resnet18(pretrained=True).cuda().eval()
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input((1, 3, size, size), dtype=torch.float32)],
        enabled_precisions={torch.float},
        workspace_size=1 << 22,
    )
    with open(f"trt/resnet18_{size}_trt.pkl", "wb") as f:
        pickle.dump(trt_model, f)
    print(f"Saved TensorRT model for {size}x{size} as pickle")
