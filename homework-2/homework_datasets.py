import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_breast_cancer
import os

# Импортируем наши кастомные модели
from homework_model_modification import LinearRegressionModified, LogisticRegressionModified, train_with_early_stopping, train_logistic_regression

# Создаем папку для данных
os.makedirs('data', exist_ok=True)


class CustomCSVDataset:
    def __init__(self, file_path: str, target_column: str,
                 normalize: bool = True, test_size: float = 0.2,
                 random_state: int = 42):
        """
        Простой кастомный класс датасета для CSV файлов

        Args:
            file_path: Путь к CSV файлу
            target_column: Название целевой колонки
            normalize: Нормализовать ли признаки
            test_size: Доля данных для тестового набора
            random_state: Случайное зерно
        """
        self.file_path = file_path
        self.target_column = target_column
        self.normalize = normalize
        self.test_size = test_size
        self.random_state = random_state

        # Объекты предобработки
        self.scaler = StandardScaler() if normalize else None

        # Загружаем и предобрабатываем данные
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """Загружает данные из CSV и применяет предобработку"""
        # Загружаем данные
        print(f"Загружаем данные из {self.file_path}")
        df = pd.read_csv(self.file_path)
        print(f"Размер данных: {df.shape}")

        # Обрабатываем пропущенные значения
        df = df.dropna()
        print(f"После удаления пропущенных значений: {df.shape}")

        # Разделяем целевую переменную и признаки
        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])

        # Убеждаемся, что бинарные метки классификации - целые числа
        if set(np.unique(y)) <= {0, 1}:
            y = y.astype(np.int64)

        # Конвертируем категориальные колонки в числовые
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

        # Нормализуем признаки если требуется
        if self.normalize and self.scaler is not None:
            X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Конвертируем в тензоры
        self.X_train = torch.tensor(X_train.values, dtype=torch.float32)
        self.X_test = torch.tensor(X_test.values, dtype=torch.float32)

        # Обрабатываем целевую переменную
        if y.dtype == 'object' or y.dtype == 'category':
            # Классификация
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            self.y_train = torch.tensor(y_train_encoded, dtype=torch.long)
            self.y_test = torch.tensor(y_test_encoded, dtype=torch.long)
            self.num_classes = 2
            print(f"Задача классификации с {self.num_classes} классами")
        else:
            # Регрессия
            self.y_train = torch.tensor(
                y_train.values, dtype=torch.float32).unsqueeze(1)
            self.y_test = torch.tensor(
                y_test.values, dtype=torch.float32).unsqueeze(1)
            self.num_classes = 1
            print("Задача регрессии")

        self.feature_names = X.columns.tolist()
        print(f"Признаков: {len(self.feature_names)}")
        print(f"Обучающий набор: {self.X_train.shape}")
        print(f"Тестовый набор: {self.X_test.shape}")

    def get_data(self):
        """Получить обработанные данные"""
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_feature_names(self):
        """Получить названия признаков"""
        return self.feature_names

    def is_classification(self):
        """Проверить, является ли это задачей классификации"""
        return self.num_classes > 1


def create_diabetes_dataset():
    """Создает датасет Diabetes для регрессии"""
    print("=== Датасет Diabetes ===")

    # Загружаем датасет
    diabetes = load_diabetes()
    if isinstance(diabetes, tuple):
        diabetes = diabetes[0]
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target

    # Сохраняем в CSV
    df.to_csv("data/diabetes.csv", index=False)

    # Создаем датасет
    dataset = CustomCSVDataset(
        file_path="data/diabetes.csv",
        target_column='target',
        normalize=True
    )

    return dataset


def create_breast_cancer_dataset():
    """Создает датасет Breast Cancer для классификации"""
    print("=== Датасет Breast Cancer ===")

    # Загружаем датасет
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target

    # Сохраняем в CSV
    df.to_csv("data/breast_cancer.csv", index=False)

    # Создаем датасет
    dataset = CustomCSVDataset(
        file_path="data/breast_cancer.csv",
        target_column='target',
        normalize=True
    )

    return dataset


def train_regression_model(dataset):
    """Обучает модель линейной регрессии"""
    print("\n=== Обучение линейной регрессии ===")

    X_train, X_test, y_train, y_test = dataset.get_data()

    # Создаем модель
    model = LinearRegressionModified(
        in_features=X_train.shape[1],
        l1_lambda=0.01,
        l2_lambda=0.1
    )

    # Обучаем модель
    history = train_with_early_stopping(
        model, X_train, y_train, X_test, y_test,
        epochs=500, lr=0.01, verbose=True
    )

    # Оцениваем
    with torch.no_grad():
        y_pred = model(X_test)
        mse = torch.mean((y_pred - y_test) ** 2).item()
        rmse = np.sqrt(mse)

    print(f"Тестовая RMSE: {rmse:.4f}")

    return model, history


def train_classification_model(dataset):
    """Обучает модель логистической регрессии"""
    print("\n=== Обучение логистической регрессии ===")

    X_train, X_test, y_train, y_test = dataset.get_data()

    # Создаем модель
    model = LogisticRegressionModified(
        in_features=X_train.shape[1],
        num_classes=2,  # Всегда 2 для бинарной классификации
        l1_lambda=0.01,
        l2_lambda=0.1
    )

    # Обучаем модель
    history = train_logistic_regression(
        model, X_train, y_train, X_test, y_test,
        epochs=500, lr=0.1, verbose=True
    )

    print("Метрики валидации:")
    for metric, value in history['val_metrics'].items():
        print(f"  {metric}: {value:.4f}")

    return model, history


def analyze_features(model, feature_names, model_type="regression"):
    """Анализирует важность признаков"""
    print(f"\n=== Важность признаков ({model_type}) ===")

    if model_type == "regression":
        weights = model.w.flatten().abs().numpy()
    else:
        weights = model.w.abs().mean(dim=1).numpy()

    # Создаем DataFrame важности
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': weights
    }).sort_values('importance', ascending=False)

    print("Топ-10 признаков:")
    print(importance_df.head(10))

    return importance_df


if __name__ == "__main__":
    print("===Задание 2: Работа с датасетами ===\n")

    # 1. Пример регрессии
    print("1. ПРИМЕР РЕГРЕССИИ")
    print("=" * 40)

    diabetes_dataset = create_diabetes_dataset()
    regression_model, regression_history = train_regression_model(
        diabetes_dataset)
    analyze_features(regression_model,
                     diabetes_dataset.get_feature_names(), "regression")

    # 2. Пример классификации
    print("\n2. ПРИМЕР КЛАССИФИКАЦИИ")
    print("=" * 40)

    cancer_dataset = create_breast_cancer_dataset()
    classification_model, classification_history = train_classification_model(
        cancer_dataset)
    analyze_features(classification_model,
                     cancer_dataset.get_feature_names(), "classification")
