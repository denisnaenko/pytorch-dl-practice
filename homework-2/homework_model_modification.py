import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import os

# Создаем папку для моделей
os.makedirs('models', exist_ok=True)


class LinearRegressionModified:
    def __init__(self, in_features: int, l1_lambda: float = 0.0, l2_lambda: float = 0.0):
        """
        Инициализирует модифицированную линейную регрессию с регуляризацией

        Args:
            in_features: Количество входных признаков
            l1_lambda: Коэффициент L1 регуляризации (Lasso)
            l2_lambda: Коэффициент L2 регуляризации (Ridge)
        """
        self.w = torch.randn(
            in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)

        # Параметры регуляризации
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # Параметры early stopping
        self.best_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        self.best_w: Optional[torch.Tensor] = None
        self.best_b: Optional[torch.Tensor] = None

        # Инициализируем градиенты
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b

    def parameters(self) -> List[torch.Tensor]:
        return [self.w, self.b]

    def zero_grad(self) -> None:
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> None:
        n = X.shape[0]
        error = y_pred - y

        # Градиент для весов
        self.dw = (X.T @ error) / n

        # Добавляем градиенты регуляризации
        if self.l1_lambda > 0:
            self.dw += self.l1_lambda * torch.sign(self.w)
        if self.l2_lambda > 0:
            self.dw += self.l2_lambda * self.w

        # Градиент для смещения (без регуляризации)
        self.db = error.mean(0)

    def step(self, lr: float) -> None:
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def compute_loss(self, X: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Вычисляет MSE loss с регуляризацией"""
        mse_loss = torch.mean((y_pred - y) ** 2)

        # Добавляем члены регуляризации
        reg_loss = 0.0
        if self.l1_lambda > 0:
            reg_loss += self.l1_lambda * torch.sum(torch.abs(self.w))
        if self.l2_lambda > 0:
            reg_loss += self.l2_lambda * torch.sum(self.w ** 2)

        return mse_loss + reg_loss

    def early_stopping_check(self, current_loss: float) -> bool:
        """
        Проверяет, нужно ли запустить early stopping

        Args:
            current_loss: Текущая валидационная ошибка

        Returns:
            bool: True если обучение должно остановиться, False иначе
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
            # Сохраняем лучшие параметры
            self.best_w = self.w.clone()
            self.best_b = self.b.clone()
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                # Восстанавливаем лучшие параметры
                if self.best_w is not None and self.best_b is not None:
                    self.w = self.best_w
                    self.b = self.best_b
                return True
            return False

    def set_early_stopping_params(self, patience: int = 10) -> None:
        """Устанавливает параметры early stopping"""
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_w = None
        self.best_b = None

    def save(self, path: str) -> None:
        """Сохраняет модель"""
        torch.save({
            'w': self.w,
            'b': self.b,
            'l1_lambda': self.l1_lambda,
            'l2_lambda': self.l2_lambda
        }, path)

    def load(self, path: str) -> None:
        """Загружает модель"""
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']
        self.l1_lambda = state.get('l1_lambda', 0.0)
        self.l2_lambda = state.get('l2_lambda', 0.0)


def train_with_early_stopping(model: LinearRegressionModified,
                              X_train: torch.Tensor, y_train: torch.Tensor,
                              X_val: torch.Tensor, y_val: torch.Tensor,
                              epochs: int = 1000, lr: float = 0.01,
                              verbose: bool = True) -> Dict[str, Any]:
    """
    Обучает модель с early stopping

    Args:
        model: Экземпляр LinearRegressionModified
        X_train, y_train: Обучающие данные
        X_val, y_val: Валидационные данные
        epochs: Максимальное количество эпох
        lr: Скорость обучения
        verbose: Выводить ли прогресс обучения

    Returns:
        dict: История обучения
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Шаг обучения
        model.zero_grad()
        y_pred_train = model(X_train)
        train_loss = model.compute_loss(X_train, y_train, y_pred_train)
        model.backward(X_train, y_train, y_pred_train)
        model.step(lr)

        # Шаг валидации
        with torch.no_grad():
            y_pred_val = model(X_val)
            val_loss = model.compute_loss(X_val, y_val, y_pred_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if verbose and epoch % 100 == 0:
            print(
                f"Эпоха {epoch}: Ошибка обучения = {train_loss:.6f}, Ошибка валидации = {val_loss:.6f}")

        # Проверка early stopping
        if model.early_stopping_check(val_loss.item()):
            if verbose:
                print(f"Early stopping сработал на эпохе {epoch}")
            break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_trained': len(train_losses)
    }


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Сигмоидная функция активации"""
    return 1 / (1 + torch.exp(-x))


class LogisticRegressionModified:
    def __init__(self, in_features: int, num_classes: int = 2, l1_lambda: float = 0.0, l2_lambda: float = 0.0):
        """
        Инициализирует модифицированную логистическую регрессию с поддержкой многоклассовой классификации

        Args:
            in_features: Количество входных признаков
            num_classes: Количество классов (2 для бинарной, >2 для многоклассовой)
            l1_lambda: Коэффициент L1 регуляризации
            l2_lambda: Коэффициент L2 регуляризации
        """
        self.in_features = in_features
        self.num_classes = num_classes

        # Для бинарной классификации
        if num_classes == 2:
            self.w = torch.randn(
                in_features, 1, dtype=torch.float32, requires_grad=False)
            self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        # Для многоклассовой классификации
        else:
            self.w = torch.randn(in_features, num_classes,
                                 dtype=torch.float32, requires_grad=False)
            self.b = torch.zeros(
                num_classes, dtype=torch.float32, requires_grad=False)

        # Параметры регуляризации
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # Параметры early stopping
        self.best_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        self.best_w: Optional[torch.Tensor] = None
        self.best_b: Optional[torch.Tensor] = None

        # Инициализируем градиенты
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Прямой проход"""
        logits = X @ self.w + self.b
        if self.num_classes == 2:
            return sigmoid(logits)
        else:
            return F.softmax(logits, dim=1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Получает предсказания классов"""
        with torch.no_grad():
            probs = self(X)
            if self.num_classes == 2:
                return (probs > 0.5).float()
            else:
                return torch.argmax(probs, dim=1)

    def parameters(self) -> List[torch.Tensor]:
        return [self.w, self.b]

    def zero_grad(self) -> None:
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> None:
        n = X.shape[0]

        if self.num_classes == 2:
            # Бинарная классификация
            error = y_pred - y
            self.dw = (X.T @ error) / n
            self.db = error.mean(0)
        else:
            # Многоклассовая классификация
            y = y.long()
            if y.min() < 0 or y.max() >= self.num_classes:
                raise ValueError(
                    f"Значения классов должны быть в [0, {self.num_classes-1}], но получены min={y.min().item()}, max={y.max().item()}")
            y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
            error = y_pred - y_one_hot
            self.dw = (X.T @ error) / n
            self.db = error.mean(0)

        # Добавляем градиенты регуляризации
        if self.l1_lambda > 0:
            self.dw += self.l1_lambda * torch.sign(self.w)
        if self.l2_lambda > 0:
            self.dw += self.l2_lambda * self.w

    def step(self, lr: float) -> None:
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def compute_loss(self, X: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Вычисляет cross-entropy loss с регуляризацией"""
        if self.num_classes == 2:
            # Бинарная cross-entropy
            loss = -torch.mean(y * torch.log(y_pred + 1e-8) +
                               (1 - y) * torch.log(1 - y_pred + 1e-8))
        else:
            # Многоклассовая cross-entropy
            y = y.long()  # Убеждаемся в правильном типе для one-hot
            num_classes = int(y.max().item()) + 1
            y_one_hot = F.one_hot(y, num_classes=num_classes).float()
            loss = -torch.mean(torch.sum(y_one_hot *
                               torch.log(y_pred + 1e-8), dim=1))

        # Добавляем члены регуляризации
        reg_loss = 0.0
        if self.l1_lambda > 0:
            reg_loss += self.l1_lambda * torch.sum(torch.abs(self.w))
        if self.l2_lambda > 0:
            reg_loss += self.l2_lambda * torch.sum(self.w ** 2)

        return loss + reg_loss

    def early_stopping_check(self, current_loss: float) -> bool:
        """Проверяет, нужно ли запустить early stopping"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
            self.best_w = self.w.clone()
            self.best_b = self.b.clone()
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                if self.best_w is not None and self.best_b is not None:
                    self.w = self.best_w
                    self.b = self.best_b
                return True
            return False

    def set_early_stopping_params(self, patience: int = 10) -> None:
        """Устанавливает параметры early stopping"""
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_w = None
        self.best_b = None

    def save(self, path: str) -> None:
        """Сохраняет модель"""
        torch.save({
            'w': self.w,
            'b': self.b,
            'num_classes': self.num_classes,
            'in_features': self.in_features,
            'l1_lambda': self.l1_lambda,
            'l2_lambda': self.l2_lambda
        }, path)

    def load(self, path: str) -> None:
        """Загружает модель"""
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']
        self.num_classes = state.get('num_classes', 2)
        self.in_features = state.get('in_features', self.w.shape[0])
        self.l1_lambda = state.get('l1_lambda', 0.0)
        self.l2_lambda = state.get('l2_lambda', 0.0)


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, y_probs: torch.Tensor) -> Dict[str, float]:
    """
    Вычисляет метрики классификации

    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        y_probs: Предсказанные вероятности

    Returns:
        dict: Словарь с метриками
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_probs_np = y_probs.cpu().numpy()

    # Precision, Recall, F1-score
    if y_probs.shape[1] == 2:  # Бинарная классификация
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='binary', zero_division='warn'
        )
        # ROC-AUC для бинарной классификации
        auc = float(roc_auc_score(y_true_np, y_probs_np[:, 1]))
    else:  # Многоклассовая классификация
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='weighted', zero_division='warn'
        )
        # ROC-AUC для многоклассовой (one-vs-rest)
        if y_probs.shape[1] > 2:
            try:
                auc = float(roc_auc_score(y_true_np, y_probs_np,
                                          multi_class='ovr', average='weighted'))
            except ValueError:
                auc = 0.0  # Fallback если AUC не может быть вычислен
        else:
            auc = 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(auc)
    }


def plot_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor,
                          class_names: Optional[List[str]] = None,
                          title: str = "Матрица ошибок", save_path: Optional[str] = None) -> None:
    """
    Строит матрицу ошибок

    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        class_names: Названия классов
        title: Заголовок графика
        save_path: Путь для сохранения графика (если None, показывается на экране)
    """
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    # Вычисляем матрицу ошибок
    cm = confusion_matrix(y_true_np, y_pred_np)

    # Создаем график
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names is not None else 'auto',
                yticklabels=class_names if class_names is not None else 'auto')
    plt.title(title)
    plt.ylabel('Истинная метка')
    plt.xlabel('Предсказанная метка')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Матрица ошибок сохранена в {save_path}")
    else:
        plt.show()


def train_logistic_regression(model: LogisticRegressionModified,
                              X_train: torch.Tensor, y_train: torch.Tensor,
                              X_val: torch.Tensor, y_val: torch.Tensor,
                              epochs: int = 1000, lr: float = 0.01,
                              verbose: bool = True) -> Dict[str, Any]:
    """
    Обучает модель логистической регрессии с early stopping

    Args:
        model: Экземпляр LogisticRegressionModified
        X_train, y_train: Обучающие данные
        X_val, y_val: Валидационные данные
        epochs: Максимальное количество эпох
        lr: Скорость обучения
        verbose: Выводить ли прогресс обучения

    Returns:
        dict: История обучения и финальные метрики
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Шаг обучения
        model.zero_grad()
        y_pred_train = model(X_train)
        train_loss = model.compute_loss(X_train, y_train, y_pred_train)
        model.backward(X_train, y_train, y_pred_train)
        model.step(lr)

        # Шаг валидации
        with torch.no_grad():
            y_pred_val = model(X_val)
            val_loss = model.compute_loss(X_val, y_val, y_pred_val)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if verbose and epoch % 100 == 0:
            print(
                f"Эпоха {epoch}: Ошибка обучения = {train_loss:.6f}, Ошибка валидации = {val_loss:.6f}")

        # Проверка early stopping
        if model.early_stopping_check(val_loss.item()):
            if verbose:
                print(f"Early stopping сработал на эпохе {epoch}")
            break

    # Вычисляем финальные метрики
    with torch.no_grad():
        y_pred_train_final = model.predict(X_train)
        y_pred_val_final = model.predict(X_val)
        y_probs_train = model(X_train)
        y_probs_val = model(X_val)

        train_metrics = compute_metrics(
            y_train, y_pred_train_final, y_probs_train)
        val_metrics = compute_metrics(y_val, y_pred_val_final, y_probs_val)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_trained': len(train_losses),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'y_pred_train': y_pred_train_final,
        'y_pred_val': y_pred_val_final,
        'y_probs_train': y_probs_train,
        'y_probs_val': y_probs_val
    }


# Пример использования
if __name__ == "__main__":
    print("=== Пример линейной регрессии ===")
    # Генерируем примерные данные
    torch.manual_seed(42)
    X = torch.randn(100, 5)
    y = X @ torch.randn(5, 1) + 0.1 * torch.randn(100, 1)

    # Разделяем данные
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Создаем модель с регуляризацией
    model = LinearRegressionModified(
        in_features=5,
        l1_lambda=0.01,  # L1 регуляризация
        l2_lambda=0.1    # L2 регуляризация
    )

    # Устанавливаем параметры early stopping
    model.set_early_stopping_params(patience=20)

    # Обучаем с early stopping
    history = train_with_early_stopping(
        model, X_train, y_train, X_val, y_val,
        epochs=1000, lr=0.01, verbose=True
    )

    print(f"Обучение завершено за {history['epochs_trained']} эпох")
    print(f"Финальные веса: {model.w.flatten()}")
    print(f"Финальное смещение: {model.b.item()}")

    # Сохраняем модель
    model.save('models/linear_regression_model.pth')
    print("Модель сохранена в models/linear_regression_model.pth")

    print("\n=== Пример логистической регрессии ===")
    # Генерируем примерные данные для бинарной классификации
    torch.manual_seed(42)
    X_binary = torch.randn(200, 3)
    # Создаем линейно разделимые данные
    y_binary = ((X_binary[:, 0] + X_binary[:, 1] -
                X_binary[:, 2]) > 0).float().unsqueeze(1)

    # Разделяем данные
    train_size = int(0.8 * len(X_binary))
    X_train_bin, X_val_bin = X_binary[:train_size], X_binary[train_size:]
    y_train_bin, y_val_bin = y_binary[:train_size], y_binary[train_size:]

    # Создаем модель бинарной классификации
    model_binary = LogisticRegressionModified(
        in_features=3,
        num_classes=2,
        l1_lambda=0.01,
        l2_lambda=0.1
    )

    # Устанавливаем параметры early stopping
    model_binary.set_early_stopping_params(patience=20)

    # Обучаем модель
    history_binary = train_logistic_regression(
        model_binary, X_train_bin, y_train_bin, X_val_bin, y_val_bin,
        epochs=1000, lr=0.1, verbose=True
    )

    print(f"\nРезультаты бинарной классификации:")
    print(f"Обучение завершено за {history_binary['epochs_trained']} эпох")
    print(f"Метрики валидации:")
    for metric, value in history_binary['val_metrics'].items():
        print(f"  {metric}: {value:.4f}")

    # Сохраняем модель
    model_binary.save('models/binary_classification_model.pth')
    print("Модель сохранена в models/binary_classification_model.pth")

    # Строим матрицу ошибок для бинарной классификации
    plot_confusion_matrix(
        y_val_bin.squeeze(),
        history_binary['y_pred_val'].squeeze(),
        class_names=['Класс 0', 'Класс 1'],
        title="Матрица ошибок бинарной классификации",
        save_path="plots/binary_confusion_matrix.png"
    )

    # Пример многоклассовой классификации
    print("\n=== Пример многоклассовой классификации ===")
    # Генерируем примерные данные для 3-классовой классификации
    X_multi = torch.randn(300, 4)
    # Создаем 3 кластера
    y_multi = torch.zeros(300, dtype=torch.long)
    y_multi[X_multi[:, 0] + X_multi[:, 1] > 0] = 1
    y_multi[X_multi[:, 2] + X_multi[:, 3] > 0] = 2

    # Разделяем данные
    train_size = int(0.8 * len(X_multi))
    X_train_multi, X_val_multi = X_multi[:train_size], X_multi[train_size:]
    y_train_multi, y_val_multi = y_multi[:train_size], y_multi[train_size:]

    # Создаем многоклассовую модель
    model_multi = LogisticRegressionModified(
        in_features=4,
        num_classes=3,
        l1_lambda=0.01,
        l2_lambda=0.1
    )

    # Обучаем модель
    history_multi = train_logistic_regression(
        model_multi, X_train_multi, y_train_multi, X_val_multi, y_val_multi,
        epochs=1000, lr=0.1, verbose=True
    )

    print(f"\nРезультаты многоклассовой классификации:")
    print(f"Обучение завершено за {history_multi['epochs_trained']} эпох")
    print(f"Метрики валидации:")
    for metric, value in history_multi['val_metrics'].items():
        print(f"  {metric}: {value:.4f}")

    # Сохраняем модель
    model_multi.save('models/multiclass_classification_model.pth')
    print("Модель сохранена в models/multiclass_classification_model.pth")

    # Строим матрицу ошибок для многоклассовой классификации
    plot_confusion_matrix(
        y_val_multi,
        history_multi['y_pred_val'],
        class_names=['Класс 0', 'Класс 1', 'Класс 2'],
        title="Матрица ошибок многоклассовой классификации",
        save_path="plots/multiclass_confusion_matrix.png"
    )
