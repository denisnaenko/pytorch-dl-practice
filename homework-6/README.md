# Домашнее задание 6: Генератор текста на базе Transformer

## Описание

Этот проект реализует генератор текста на русском языке на базе архитектуры Transformer (декодер). Модель обучается на большом корпусе новостных текстов и может генерировать продолжения фраз в авторегрессивном режиме, поддерживает beam search.

---

## Используемый датасет

Для обучения используется открытый корпус новостных текстов на русском языке — [russian_news_corpus](https://github.com/maxoodf/russian_news_corpus):
- Более 1,5 млн статей российских СМИ за 2016–2017 годы
- ~4,5 ГБ текста, ~360 млн слов

### Как скачать корпус

```bash
git clone https://github.com/maxoodf/russian_news_corpus.git
cd russian_news_corpus
cat russian_news.txt.bz2_a* | bzip2 -d > russian_news.txt
```
Файл `russian_news.txt` используйте для обучения.

---

## Структура проекта

```
homework-6/
├── chat.py                  # Чат-бот для генерации текста (интерфейс)
├── generator_transformer.py # Реализация модели GeneratorTransformer
├── train.py                 # Скрипт для обучения модели
├── train_tokenizer.py       # Скрипт для обучения BPE-токенизатора
├── tokenizer/               # Каталог для файлов токенизатора
│   └── russian_tokenizer.json # (не хранится в git)
├── model/                   # Каталог для чекпоинтов модели и кэша датасета
│   └── checkpoint_fullcorpus.pt (не хранится в git)
├── russian_news_corpus/     # Каталог с исходным корпусом (не хранится в git)
└── README.md                # Описание проекта
```

---

## Как запустить

1. **Установите зависимости:**
   ```bash
   pip install torch tokenizers tqdm
   ```

2. **Обучите токенизатор:**
   ```bash
   python train_tokenizer.py
   ```
   Файл `tokenizer/russian_tokenizer.json` появится автоматически.

3. **Обучите модель:**
   ```bash
   python train.py
   ```
   После обучения появится файл чекпоинта, `model/checkpoint_fullcorpus.pt`.

4. **Запустите чат-бот:**
   ```bash
   python chat.py
   ```
   Введите фразу, выберите режим генерации (1 — обычный, 2 — beam search).

---
