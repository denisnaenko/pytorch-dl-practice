# Домашнее задание 4: Сверточные сети

### Структура проекта

```bash
homework-4/
├── homework_cnn_architecture_analysis.py      # Анализ архитектур CNN (глубина, ядра)
├── homework_cnn_vs_fc_comparison.py           # Сравнение CNN и FC на MNIST и CIFAR-10
├── homework_custom_layers_experiments.py      # Эксперименты с кастомными слоями и блоками
├── models
│   ├── cnn_models.py                         # Реализации различных CNN (в т.ч. residual, attention)
│   ├── custom_layers.py                      # Кастомные слои и блоки (например, attention, pooling)
│   └── fc_models.py                          # Реализации полносвязных сетей
├── plots
├── README.md
├── REPORT.md
├── results
│   ├── architecture_analysis
│   ├── cifar_comparison
│   └── mnist_comparison
└── utils
    ├── comparison_utils.py                   # Утилиты для сравнения и анализа моделей
    ├── training_utils.py                     # Функции обучения, подсчета параметров, инференса
    └── visualization_utils.py                # Визуализация: графики, confusion matrix, feature maps
```

## Основные сценарии использования

- **Сравнение моделей:**
  - `homework_cnn_vs_fc_comparison.py` — запуск экспериментов по сравнению FC и различных CNN на MNIST и CIFAR-10.
- **Анализ архитектур:**
  - `homework_cnn_architecture_analysis.py` — исследование влияния глубины и размера ядра на качество моделей.
- **Эксперименты с кастомными слоями:**
  - `homework_custom_layers_experiments.py` — тестирование собственных слоев, блоков и их влияния на обучение.

## Модули и утилиты

- **models/** — все реализации моделей (CNN, FC, кастомные блоки).
- **utils/** — вспомогательные функции для обучения, визуализации, анализа.
- **results/** — сохраненные результаты экспериментов (csv, pkl, отчеты).
- **plots/** — все графики и визуализации, автоматически генерируемые скриптами.

## Воспроизводимость

- Все эксперименты запускаются из соответствующих .py-файлов.
- Результаты и графики сохраняются автоматически в папки `results/` и `plots/`.

---