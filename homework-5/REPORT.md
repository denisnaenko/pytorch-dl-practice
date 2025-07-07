# Отчет по домашнему заданию 5

## 1. Стандартные аугментации torchvision

Для 5 изображений из разных классов были применены стандартные аугментации: горизонтальное отражение, кроп, изменение цвета, поворот, grayscale. Ниже приведены примеры (оригинал, отдельные аугментации, все вместе):

![Гароу стандартные](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/augs/%D0%93%D0%B0%D1%80%D0%BE%D1%83_augs.png)
![Генос стандартные](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/augs/%D0%93%D0%B5%D0%BD%D0%BE%D1%81_augs.png)

**Выводы:**
- Стандартные аугментации позволяют существенно разнообразить обучающую выборку, что помогает бороться с переобучением.
- Некоторые аугментации (например, сильный поворот или grayscale) могут искажать важные признаки класса, поэтому их силу и вероятность применения стоит подбирать с учетом специфики задачи.

## 2. Кастомные аугментации

Реализованы кастомные аугментации: размытие, перспектива, яркость/контраст. Примеры результатов:

![Гароу кастомные](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/augs/custom_%D0%93%D0%B0%D1%80%D0%BE%D1%83_augs.png)
![Генос кастомные](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/augs/custom_%D0%93%D0%B5%D0%BD%D0%BE%D1%81_augs.png)

**Выводы:**
- Кастомные аугментации позволяют имитировать реальные искажения (размытие, перспективные искажения, изменение освещенности), что делает модель более устойчивой к шуму и реальным условиям.
- Визуальное сравнение с готовыми аугментациями показывает, что кастомные методы могут быть не менее эффективны, а иногда и более релевантны для конкретной задачи.

## 3. Анализ датасета

- Количество изображений по классам, минимальный/максимальный/средний размер, средняя площадь — см. файл [dataset_stats.txt](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/analysis/dataset_stats.txt)
- Гистограмма по классам:

![Гистограмма по классам](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/analysis/class_hist.png)

- Распределение размеров:

![Распределение размеров](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/analysis/size_hist.png)

- Scatter plot ширина/высота:

![Scatter plot](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/analysis/wh_scatter.png)

**Выводы:**
- Размеры изображений варьируются, что требует приведения к единому размеру на этапе препроцессинга.
- Средний размер изображений и их разброс важны для выбора архитектуры и параметров модели.

## 4. Pipeline аугментаций

Реализован класс AugmentationPipeline с методами add/remove/apply/get. Примеры применения разных конфигураций:

- Light:
![Light pipeline](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/pipeline/light_Гароу_1a889943f42991fb690023b96874d47e.jpg)
- Medium:
![Medium pipeline](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/pipeline/medium_Генос_6b4891f65722dec908bf1bbc5cf1ca46.jpg)
- Heavy:
![Heavy pipeline](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/pipeline/heavy_Гароу_1a889943f42991fb690023b96874d47e.jpg)

**Выводы:**
- Гибкая настройка пайплайна аугментаций позволяет адаптировать сложность аугментаций под задачу и качество данных.
- Сильные аугментации (heavy) могут быть полезны для борьбы с переобучением, но при избыточной силе могут ухудшать сходимость модели.


## 5. Эксперимент с размерами

Проведен эксперимент с размерами 64, 128, 224, 512 px. Замерялись время и память на 100 изображениях.

- Время vs размер:
![Time vs size](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/resize_exp/time_vs_size.png)
- Память vs размер:
![Mem vs size](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/resize_exp/mem_vs_size.png)

Таблица результатов: [resize_experiment.txt](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/resize_exp/resize_experiment.txt)

**Возможная причина аномалии на графике памяти:**
- На графике "Память vs Размер изображений" наблюдается резкое падение и последующий рост потребления памяти. Это может быть связано с особенностями измерения памяти в Python и поведением операционной системы: память может освобождаться не сразу, часть данных кэшируется, а также возможны колебания из-за работы сборщика мусора и других фоновых процессов. Но если исключить этот выброс, то можно видеть, что потребление памяти возрастает c увеличением размера изображений.

**Выводы:**
- Время обработки и потребление памяти возрастают с увеличением размера изображения. Это важно учитывать при выборе input size для обучения моделей, особенно при ограниченных ресурсах.
- Размер 224x224 является хорошим компромиссом между качеством и производительностью для большинства задач классификации.
- Для задач, где важны детали, можно использовать большие размеры, но это требует увеличения вычислительных ресурсов.

## 6. Дообучение предобученной модели

Для задачи классификации был дообучен ResNet18. Последний слой заменен на количество классов датасета. Обучение велось 5 эпох, результаты:

- Loss по эпохам:
![Loss curve](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/finetune/loss_curve.png)
- Accuracy по эпохам:
![Accuracy curve](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/finetune/acc_curve.png)

Веса модели сохранены: [finetuned_resnet18.pth](https://github.com/denisnaenko/pytorch-dl-practice/blob/main/homework-5/results/finetune/finetuned_resnet18.pth)

**Выводы:**
- Модель быстро выходит на высокое качество на train, но на val наблюдается переобучение (разрыв между loss/acc). Это связано с малым размером валидационной выборки взятого для данного эксперимента.
- Для улучшения обобщающей способности модели можно использовать более сильные аугментации, регуляризацию, а также балансировку классов.
- Использование предобученных моделей существенно ускоряет сходимость и повышает итоговое качество по сравнению с обучением с нуля.

---

**Все скрипты, графики и результаты доступны в папке homework-5/results/**
