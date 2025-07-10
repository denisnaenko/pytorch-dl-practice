import torch
import torchvision.models as models
import os

os.makedirs("weights", exist_ok=True)

for size in [224, 256, 384, 512]:
    model = models.resnet18(pretrained=True)
    model.eval()
    dummy = torch.randn(1, 3, size, size)

    # Сохраняем веса
    torch.save(model.state_dict(), f"weights/best_resnet18_{size}.pth")
    print(f"Saved weights for {size}x{size}")
