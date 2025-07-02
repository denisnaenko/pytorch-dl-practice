# Домашнее задание 2: Линейная и логистическая регрессия

## Структура проекта

```
homework-2/
├── data/                           # Датасеты в формате CSV
│   ├── diabetes.csv               # Датасет для регрессии
│   └── breast_cancer.csv          # Датасет для классификации
├── models/                        # Сохраненные модели
│   ├── linear_regression_model.pth
│   ├── binary_classification_model.pth
│   └── multiclass_classification_model.pth
├── plots/                         # Графики и визуализации
│   ├── hyperparams_classification.png
│   ├── hyperparams_regression.png
│   ├── feature_engineering_classification.png
│   ├── feature_engineering_regression.png
│   ├── binary_confusion_matrix.png
│   └── multiclass_confusion_matrix.png
├── homework_model_modification.py  # Основные модели
├── homework_datasets.py           # Обработка данных
├── homework_experiments.py        # Система экспериментов
└── README.md                      # Документация
```

## Файлы проекта

### homework_model_modification.py

Основной файл с реализацией модифицированных моделей машинного обучения.

#### Классы моделей

**LinearRegressionModified**
- Линейная регрессия с L1/L2 регуляризацией
- Early stopping для предотвращения переобучения
- Методы: `__call__`, `parameters`, `zero_grad`, `backward`, `step`, `compute_loss`
- Параметры: `in_features`, `l1_lambda`, `l2_lambda`

**LogisticRegressionModified**
- Логистическая регрессия с поддержкой бинарной и многоклассовой классификации
- L1/L2 регуляризация и early stopping
- Методы: `__call__`, `predict`, `parameters`, `zero_grad`, `backward`, `step`, `compute_loss`
- Параметры: `in_features`, `num_classes`, `l1_lambda`, `l2_lambda`

#### Функции обучения

**train_with_early_stopping**
- Обучение линейной регрессии с early stopping
- Возвращает историю обучения (train_losses, val_losses, epochs_trained)

**train_logistic_regression**
- Обучение логистической регрессии с метриками
- Возвращает историю обучения и метрики классификации

#### Функции анализа

**compute_metrics**
- Вычисление метрик классификации: precision, recall, F1-score, ROC-AUC
- Поддержка бинарной и многоклассовой классификации

**plot_confusion_matrix**
- Построение матрицы ошибок с использованием seaborn
- Сохранение в папку plots/ или отображение на экране

### homework_datasets.py

Файл для работы с датасетами и их предобработки.

#### Класс CustomCSVDataset

**Основные возможности:**
- Загрузка данных из CSV файлов
- Обработка пропущенных значений
- Нормализация признаков
- Кодирование категориальных переменных
- Автоматическое определение типа задачи (регрессия/классификация)
- Разделение на обучающую и тестовую выборки

**Методы:**
- `get_data()` - получение обработанных данных
- `get_feature_names()` - получение названий признаков
- `is_classification()` - проверка типа задачи

#### Функции создания датасетов

**create_diabetes_dataset**
- Создание датасета Diabetes для задачи регрессии
- Загрузка из sklearn.datasets
- Сохранение в data/diabetes.csv

**create_breast_cancer_dataset**
- Создание датасета Breast Cancer для задачи классификации
- Загрузка из sklearn.datasets
- Сохранение в data/breast_cancer.csv

#### Функции обучения и анализа

**train_regression_model**
- Обучение модели линейной регрессии
- Вычисление RMSE
- Вывод результатов обучения

**train_classification_model**
- Обучение модели логистической регрессии
- Вычисление метрик классификации
- Вывод результатов обучения

**analyze_features**
- Анализ важности признаков
- Построение графика важности признаков
- Вывод топ-10 важных признаков

### homework_experiments.py

Система экспериментов для исследования гиперпараметров и инженерных признаков.

#### Функции экспериментов

**run_lr_batchsize_experiment**
- Эксперименты с различными скоростями обучения, размерами батчей и оптимизаторами
- Поддержка SGD, Adam, RMSprop
- Сохранение результатов в plots/hyperparams_{task}.png
- Возвращает DataFrame с результатами

**compare_feature_engineering**
- Сравнение различных методов инженерных признаков
- Полиномиальные признаки (степень 2)
- Признаки взаимодействия (попарные произведения)
- Статистические признаки (среднее, дисперсия)
- Сохранение результатов в plots/feature_engineering_{task}.png

#### Функции инженерных признаков

**add_polynomial_features**
- Добавление полиномиальных признаков степени 2
- Использование sklearn.PolynomialFeatures

**add_interaction_features**
- Добавление попарных признаков взаимодействия
- Произведение всех пар признаков

**add_statistical_features**
- Добавление статистических признаков
- Среднее значение и дисперсия по признакам


## Графики и визуализации

### plots/hyperparams_{task}.png
- График поиска гиперпараметров
- По оси X: скорость обучения
- По оси Y: финальная валидационная ошибка
- Разные линии для различных комбинаций оптимизатор/размер батча
- Сетка для лучшей читаемости

### plots/feature_engineering_{task}.png
- Сравнение методов инженерных признаков
- Столбчатая диаграмма с четырьмя категориями: base, poly, interaction, stat
- По оси Y: финальная валидационная ошибка
- Цветовая схема для различия методов

### plots/binary_confusion_matrix.png
- Матрица ошибок для бинарной классификации
- Тепловая карта с числовыми значениями
- Подписи осей: "Истинная метка", "Предсказанная метка"
- Цветовая схема Blues

### plots/multiclass_confusion_matrix.png
- Матрица ошибок для многоклассовой классификации
- Тепловая карта для трех классов
- Подписи классов: "Класс 0", "Класс 1", "Класс 2"

## Запуск проекта

### Основные эксперименты
```bash
python homework_experiments.py
```

### Тестирование моделей
```bash
python homework_model_modification.py
```

### Работа с датасетами
```bash
python homework_datasets.py
```

## Требования

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
