""" Задание 1: Сравнение CNN и полносвязных сетей

# 1.1 Сравнение на MNIST

 Сравните производительность на MNIST:
  - Полносвязная сеть (3-4 слоя)
  - Простая CNN (2-3 conv слоя)
  - CNN c Residual Block
 
 Для каждого варианта:
  - Обучите модель c одинаковыми гиперпараметрами
  - Сравните точность на train и test множествах
  - Измерьте время обучения и инференса
  - Визуализируйте кривые обучения
  - Проанализируйте количество параметров

# 1.2 Сравнение на CIFAR-10

 Сравните производительность на CIFAR-10:
  - Полносвязная сеть (глубокая)
  - CNN c Residual блоками
  - CNN c регуляризацией и Residual блоками
 
 Для каждого варианта:
  - Обучите модель c одинаковыми гиперпараметрами
  - Сравните точность и время обучения
  - Проанализируйте переобучение
  - Визуализируйте confusion matrix
  - Исследуйте градиенты (gradient flow)
"""
