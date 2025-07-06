import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Базовый Residual блок"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckResidualBlock(nn.Module):
    """Bottleneck Residual блок"""

    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                               3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SimpleCNN(nn.Module):
    """Простая CNN для MNIST"""

    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNWithResidual(nn.Module):
    """CNN с Residual блоками"""

    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, 2)
        self.res3 = ResidualBlock(64, 64)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CIFARCNN(nn.Module):
    """CNN для CIFAR-10"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # Динамически определяем размер входа в fc1
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            flat_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeepCNN(nn.Module):
    """Глубокая CNN"""

    def __init__(self, input_channels=3, num_classes=10, input_size=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        # Динамически определяем размер входа в fc1
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))
            flat_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNWithDifferentKernels(nn.Module):
    """CNN с разными размерами ядер"""

    def __init__(self, input_channels=1, num_classes=10, kernel_sizes=[3, 5, 7]):
        super().__init__()
        # Все conv слои принимают на вход input_channels
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(input_channels, 32, kernel_size, padding=kernel_size//2)
            for kernel_size in kernel_sizes
        ])
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # Динамически определяем размер входа в fc1
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 28, 28)
            conv_outputs = [F.relu(conv(dummy)) for conv in self.conv_layers]
            x = torch.cat(conv_outputs, dim=1)
            x = self.pool(x)
            flat_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Применяем свертки с разными ядрами параллельно
        conv_outputs = []
        for conv in self.conv_layers:
            conv_outputs.append(F.relu(conv(x)))

        # Объединяем результаты
        x = torch.cat(conv_outputs, dim=1)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class WideResNet(nn.Module):
    """Wide Residual Network"""

    def __init__(self, input_channels=3, num_classes=10, width_factor=4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16 * width_factor, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16 * width_factor)

        # Wide residual блоки
        self.layer1 = self._make_layer(16 * width_factor, 32 * width_factor, 2)
        self.layer2 = self._make_layer(32 * width_factor, 64 * width_factor, 2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * width_factor, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(WideResidualBlock(in_channels, out_channels, stride))
        layers.append(WideResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class WideResidualBlock(nn.Module):
    """Wide Residual блок"""

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNNWithAttention(nn.Module):
    """CNN с механизмом внимания"""

    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        # Attention механизм
        self.attention = nn.MultiheadAttention(
            256, num_heads=8, batch_first=True)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Применяем attention
        batch_size, channels, height, width = x.size()
        x_flat = x.view(batch_size, channels, -1).transpose(1,
                                                            # (batch, h*w, channels)
                                                            2)
        attn_output, _ = self.attention(x_flat, x_flat, x_flat)
        x = attn_output.transpose(1, 2).view(
            batch_size, channels, height, width)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_cnn_model(model_type, input_channels, num_classes, **kwargs):
    """Фабричная функция для создания CNN моделей"""
    if model_type == 'simple':
        return SimpleCNN(input_channels=input_channels, num_classes=num_classes, **kwargs)
    elif model_type == 'residual':
        return CNNWithResidual(input_channels=input_channels, num_classes=num_classes, **kwargs)
    elif model_type == 'cifar':
        return CIFARCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'deep':
        # Передаём input_size если есть, иначе по умолчанию 32 (CIFAR-10)
        input_size = kwargs.pop('input_size', 32)
        return DeepCNN(input_channels=input_channels, num_classes=num_classes, input_size=input_size, **kwargs)
    elif model_type == 'different_kernels':
        return CNNWithDifferentKernels(input_channels=input_channels, num_classes=num_classes, **kwargs)
    elif model_type == 'wide_resnet':
        return WideResNet(input_channels=input_channels, num_classes=num_classes, **kwargs)
    elif model_type == 'attention':
        return CNNWithAttention(input_channels=input_channels, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
