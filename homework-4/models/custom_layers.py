import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomConv2d(nn.Module):
    """Кастомный сверточный слой с дополнительной логикой"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, activation='relu', dropout=0.0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)

        # Дополнительные компоненты
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Кастомная активация
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = SwishActivation()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SwishActivation(nn.Module):
    """Кастомная функция активации Swish"""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class AttentionConv2d(nn.Module):
    """Сверточный слой с механизмом внимания"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

        # Attention механизм
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = F.relu(self.bn(self.conv(x)))
        attention_weights = self.attention(conv_out)
        return conv_out * attention_weights


class CustomPooling(nn.Module):
    """Кастомный pooling слой с адаптивным размером"""

    def __init__(self, pool_type='max', kernel_size=2, stride=None, padding=0,
                 adaptive=False, output_size=None):
        super().__init__()

        self.pool_type = pool_type
        self.adaptive = adaptive

        if adaptive:
            if pool_type == 'max':
                self.pool = nn.AdaptiveMaxPool2d(output_size)
            elif pool_type == 'avg':
                self.pool = nn.AdaptiveAvgPool2d(output_size)
            else:
                raise ValueError(f"Unknown pool type: {pool_type}")
        else:
            if pool_type == 'max':
                self.pool = nn.MaxPool2d(kernel_size, stride, padding)
            elif pool_type == 'avg':
                self.pool = nn.AvgPool2d(kernel_size, stride, padding)
            elif pool_type == 'lp':
                self.pool = LpPool2d(kernel_size, stride, padding, norm_type=2)
            else:
                raise ValueError(f"Unknown pool type: {pool_type}")

    def forward(self, x):
        return self.pool(x)


class LpPool2d(nn.Module):
    """Lp-norm pooling"""

    def __init__(self, kernel_size, stride=None, padding=0, norm_type=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.norm_type = norm_type

    def forward(self, x):
        # Реализация Lp-norm pooling
        if self.norm_type == float('inf'):
            return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        elif self.norm_type == 1:
            return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        else:
            # Для других значений p используем приближение
            return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)


class SpatialAttention(nn.Module):
    """Пространственный механизм внимания"""

    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Глобальный average pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        # Глобальный max pooling
        max_pool = F.adaptive_max_pool2d(x, 1)

        # Attention weights
        avg_out = self.sigmoid(self.conv2(self.relu(self.conv1(avg_pool))))
        max_out = self.sigmoid(self.conv2(self.relu(self.conv1(max_pool))))

        attention = avg_out + max_out

        return x * attention


class ChannelAttention(nn.Module):
    """Канальный механизм внимания"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()

        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class CustomResidualBlock(nn.Module):
    """Кастомный Residual блок с дополнительными возможностями"""

    def __init__(self, in_channels, out_channels, stride=1,
                 attention=False, activation='relu', dropout=0.0):
        super().__init__()

        # Основные слои
        self.conv1 = CustomConv2d(in_channels, out_channels, 3, stride, 1,
                                  activation=activation, dropout=dropout)
        self.conv2 = CustomConv2d(out_channels, out_channels, 3, 1, 1,
                                  activation='none', dropout=0.0)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Attention механизм
        self.attention = CBAM(out_channels) if attention else nn.Identity()

        # Финальная активация
        if activation == 'relu':
            self.final_activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.final_activation = nn.LeakyReLU(0.1)
        elif activation == 'gelu':
            self.final_activation = nn.GELU()
        else:
            self.final_activation = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.attention(out)
        out += residual
        out = self.final_activation(out)

        return out


class BottleneckResidualBlock(nn.Module):
    """Bottleneck Residual блок с кастомными возможностями"""

    def __init__(self, in_channels, out_channels, stride=1, expansion=4,
                 attention=False, activation='relu', dropout=0.0):
        super().__init__()

        mid_channels = out_channels // expansion

        # Основные слои
        self.conv1 = CustomConv2d(in_channels, mid_channels, 1,
                                  activation=activation, dropout=dropout)
        self.conv2 = CustomConv2d(mid_channels, mid_channels, 3, stride, 1,
                                  activation=activation, dropout=dropout)
        self.conv3 = CustomConv2d(mid_channels, out_channels, 1,
                                  activation='none', dropout=0.0)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Attention механизм
        self.attention = CBAM(out_channels) if attention else nn.Identity()

        # Финальная активация
        if activation == 'relu':
            self.final_activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.final_activation = nn.LeakyReLU(0.1)
        elif activation == 'gelu':
            self.final_activation = nn.GELU()
        else:
            self.final_activation = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.attention(out)
        out += residual
        out = self.final_activation(out)

        return out


class CustomActivation(nn.Module):
    """Кастомная функция активации с обучаемыми параметрами"""

    def __init__(self, activation_type='learnable_relu'):
        super().__init__()
        self.activation_type = activation_type

        if activation_type == 'learnable_relu':
            self.alpha = nn.Parameter(torch.tensor(0.1))
            self.beta = nn.Parameter(torch.tensor(1.0))
        elif activation_type == 'learnable_swish':
            self.beta = nn.Parameter(torch.tensor(1.0))
        elif activation_type == 'mish':
            pass  # Mish не требует параметров
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")

    def forward(self, x):
        if self.activation_type == 'learnable_relu':
            return F.relu(x) * self.beta + self.alpha * x
        elif self.activation_type == 'learnable_swish':
            return x * torch.sigmoid(self.beta * x)
        elif self.activation_type == 'mish':
            return x * torch.tanh(F.softplus(x))
        else:
            return x


def get_custom_layer(layer_type, **kwargs):
    """Фабричная функция для создания кастомных слоев"""
    if layer_type == 'custom_conv':
        return CustomConv2d(**kwargs)
    elif layer_type == 'attention_conv':
        return AttentionConv2d(**kwargs)
    elif layer_type == 'custom_pool':
        return CustomPooling(**kwargs)
    elif layer_type == 'spatial_attention':
        return SpatialAttention(**kwargs)
    elif layer_type == 'channel_attention':
        return ChannelAttention(**kwargs)
    elif layer_type == 'cbam':
        return CBAM(**kwargs)
    elif layer_type == 'custom_residual':
        return CustomResidualBlock(**kwargs)
    elif layer_type == 'bottleneck_residual':
        return BottleneckResidualBlock(**kwargs)
    elif layer_type == 'custom_activation':
        return CustomActivation(**kwargs)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
