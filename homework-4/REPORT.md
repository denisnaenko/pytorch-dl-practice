# Отчет по экспериментам

## 1. Сравнение моделей на MNIST

### Основные результаты

| Модель         | Лучшая точность | Финальная точность | Параметры | Время обучения |
|---------------|-----------------|--------------------|-----------|---------------|
| FC_Simple     | 0.9842          | 0.9827             | 0.57M     | 106.21s       |
| FC_Deep       | 0.9840          | 0.9832             | 1.50M     | 117.43s       |
| CNN_Simple    | 0.9933          | 0.9921             | 0.42M     | 119.55s       |
| CNN_Residual  | 0.9956          | 0.9918             | 0.16M     | 217.61s       |

- Сверточные сети (CNN) демонстрируют более высокую точность по сравнению с полносвязными (FC), при этом имеют меньшее число параметров.
- Остаточная CNN достигает наилучшей точности (0.9956) при минимальном числе параметров (0.16M), что свидетельствует о высокой эффективности архитектуры.
- FC-сети уступают по точности и требуют больше параметров для достижения сопоставимых результатов.

#### Визуализация:
- ![Сравнение точности моделей](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-4/plots/mnist_comparison/accuracy_comparison.png)
- ![Сравнение потерь моделей](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-4/plots/mnist_comparison/loss_comparison.png)
- ![Сравнение числа параметров](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-4/plots/mnist_comparison/parameters_comparison.png)
- ![Сравнение времени инференса](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-4/plots/mnist_comparison/inference_time_comparison.png)


## 2. Сравнение моделей на CIFAR-10

### Основные результаты

| Модель        | Лучшая точность | Финальная точность | Параметры | Время обучения |
|--------------|-----------------|--------------------|-----------|---------------|
| FC_Deep      | 0.5838          | 0.5813             | 3.84M     | 170.46s       |
| CNN_CIFAR    | 0.7661          | 0.7586             | 0.62M     | 178.07s       |
| CNN_Deep     | 0.8244          | 0.8188             | 2.61M     | 285.69s       |
| CNN_Residual | 0.8230          | 0.8205             | 0.16M     | 296.24s       |

- На CIFAR-10 преимущество сверточных архитектур становится еще более выраженным: FC-сеть значительно уступает по точности.
- Глубокие и остаточные CNN показывают лучшие результаты, при этом остаточная сеть остается самой компактной.
- Наблюдается рост переобучения у глубоких моделей, что видно по разнице между train и test accuracy.

#### Визуализация:
- ![Сравнение точности моделей](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-4/plots/cifar_comparison/accuracy_comparison.png)
- ![Сравнение потерь моделей](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-4/plots/cifar_comparison/loss_comparison.png)
- ![Сравнение числа параметров](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-4/plots/cifar_comparison/parameters_comparison.png)
- ![Сравнение времени инференса](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-4/plots/cifar_comparison/inference_time_comparison.png)


## 3. Анализ архитектур: глубина и размер ядра

### Влияние глубины

| Модель         | Лучшая точность | Параметры |
|---------------|-----------------|-----------|
| cnn_2_layers  | 0.9918          | 0.42M     |
| cnn_4_layers  | 0.9939          | 1.82M     |
| cnn_6_layers  | 0.9938          | 1.82M     |
| cnn_residual  | 0.9950          | 0.16M     |

- Увеличение глубины с 2 до 4 слоев приводит к росту точности, однако дальнейшее увеличение (до 6 слоев) не дает значимого прироста.
- Остаточная архитектура позволяет достичь максимальной точности при минимальном числе параметров.

#### Визуализация:
- ![Точность в зависимости от глубины](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-4/plots/architecture_analysis/depth_accuracy.png)
- ![Параметры в зависимости от глубины](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-4/plots/architecture_analysis/depth_params.png)

### Влияние размера ядра

| Модель        | Лучшая точность | Параметры |
|--------------|-----------------|-----------|
| kernel_3x3   | 0.9871          | 0.80M     |
| kernel_5x5   | 0.9895          | 0.81M     |
| kernel_7x7   | 0.9918          | 0.81M     |
| kernel_combo | 0.9854          | 1.61M     |

- Увеличение размера ядра с 3x3 до 7x7 приводит к небольшому росту точности, однако комбинированные ядра не дают дополнительного выигрыша.
- Оптимальным с точки зрения баланса точности и числа параметров является использование 5x5 или 7x7 ядер.

#### Визуализация:
- ![Точность в зависимости от размера ядра](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-4/plots/architecture_analysis/kernel_accuracy.png)
- ![Параметры в зависимости от размера ядра](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-4/plots/architecture_analysis/kernel_params.png)


## 4. Комплексный анализ и выводы

- Сверточные сети существенно превосходят полносвязные по точности и эффективности на изображениях.
- Остаточные архитектуры позволяют строить более глубокие сети без потери качества и с меньшим числом параметров.
- Для MNIST и CIFAR-10 оптимальными являются компактные сверточные архитектуры с остаточными связями.
- Увеличение глубины и размера ядра имеет смысл до определенного предела, после чего наблюдается эффект насыщения.
- Время инференса и число параметров критически важны для практического применения моделей.

---