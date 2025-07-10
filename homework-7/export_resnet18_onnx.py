import os

import torch
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
    )
    print(f"Exported ONNX for {size}x{size}")
