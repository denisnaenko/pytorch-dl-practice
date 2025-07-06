import torch.nn as nn
import torch.nn.functional as F


class SimpleFC(nn.Module):
    """Простая полносвязная сеть для MNIST"""

    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout=0.25):
        super().__init__()
        layers = []

        # Входной слой
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Скрытые слои
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Выходной слой
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)


class DeepFC(nn.Module):
    """Глубокая полносвязная сеть для CIFAR-10"""

    def __init__(self, input_size=3072, hidden_sizes=[1024, 512, 256, 128], num_classes=10, dropout=0.3):
        super().__init__()
        layers = []

        # Входной слой
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(dropout))

        # Скрытые слои
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.Dropout(dropout))

        # Выходной слой
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)


class WideFC(nn.Module):
    """Широкая полносвязная сеть"""

    def __init__(self, input_size=784, hidden_sizes=[1024, 1024, 1024], num_classes=10, dropout=0.2):
        super().__init__()
        layers = []

        # Входной слой
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Скрытые слои
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Выходной слой
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)


class ResidualFC(nn.Module):
    """Полносвязная сеть с residual связями"""

    def __init__(self, input_size=784, hidden_size=512, num_layers=4, num_classes=10, dropout=0.2):
        super().__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)

        # Residual блоки
        self.residual_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.residual_layers.append(ResidualFCBlock(hidden_size, dropout))

        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.input_layer(x))

        for layer in self.residual_layers:
            x = layer(x)

        return self.output_layer(x)


class ResidualFCBlock(nn.Module):
    """Residual блок для полносвязной сети"""

    def __init__(self, hidden_size, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out += residual  # Residual connection
        return F.relu(out)


class FCWithAttention(nn.Module):
    """Полносвязная сеть с механизмом внимания"""

    def __init__(self, input_size=784, hidden_sizes=[512, 256], num_classes=10, dropout=0.25):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        # Основные слои
        self.layers = nn.ModuleList()
        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        # Attention механизм
        self.attention = nn.MultiheadAttention(
            hidden_sizes[-1], num_heads=8, batch_first=True)

        # Выходной слой
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten

        # Проходим через основные слои
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
            x = self.dropout(x)

        # Применяем attention (рассматриваем каждый элемент как отдельный токен)
        x = x.unsqueeze(1)  # Добавляем размерность последовательности
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)  # Убираем размерность последовательности

        return self.output_layer(x)


def get_fc_model(model_type, input_size, num_classes, **kwargs):
    """Фабричная функция для создания полносвязных моделей"""
    if model_type == 'simple':
        return SimpleFC(input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_type == 'deep':
        return DeepFC(input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_type == 'wide':
        return WideFC(input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_type == 'residual':
        return ResidualFC(input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_type == 'attention':
        return FCWithAttention(input_size=input_size, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
