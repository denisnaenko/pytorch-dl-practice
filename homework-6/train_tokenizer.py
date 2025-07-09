# type: ignore
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

corpus_path = "russian_news_corpus/russian_news.txt"

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=20000, special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"]
)

with open(corpus_path, encoding="utf-8", errors="ignore") as f:
    lines = [line.strip() for line in f if line.strip()]

tokenizer.train_from_iterator(lines, trainer)
tokenizer.save("tokenizer/russian_tokenizer.json")

print("Tokenizer saved to tokenizer/russian_tokenizer.json")
