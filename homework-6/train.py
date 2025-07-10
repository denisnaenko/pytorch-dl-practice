import os
import time

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from generator_transformer import GeneratorTransformer


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=192, cache_path=None):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples = []

        if cache_path and os.path.exists(cache_path):
            print(f"[TextDataset] Loading cached dataset from {cache_path} ...")
            self.samples = torch.load(cache_path)
            print(f"[TextDataset] Loaded {len(self.samples)} samples from cache.")
            return

        print(f"[TextDataset] Loading and tokenizing corpus from {file_path} ...")
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line in tqdm(f, desc="Corpus lines processed"):
                ids = tokenizer.encode(line.strip()).ids
                if len(ids) < 2:
                    continue
                ids = (
                    [tokenizer.token_to_id("<bos>")]
                    + ids
                    + [tokenizer.token_to_id("<eos>")]
                )
                for j in range(0, len(ids) - max_length, max_length):
                    chunk = ids[j : j + max_length]
                    if len(chunk) < max_length:
                        chunk += [tokenizer.token_to_id("<pad>")] * (
                            max_length - len(chunk)
                        )
                    self.samples.append(chunk)
        print(f"[TextDataset] Total samples: {len(self.samples)}")
        if cache_path:
            torch.save(self.samples, cache_path)
            print(f"[TextDataset] Saved tokenized dataset to {cache_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx][:-1])
        y = torch.tensor(self.samples[idx][1:])
        return x, y


def train():
    max_length = 128
    batch_size = 2
    num_epochs = 5
    d_model = 128
    nhead = 4
    num_layers = 2

    tokenizer = Tokenizer.from_file("tokenizer/russian_tokenizer.json")
    dataset = TextDataset(
        "russian_news_corpus/russian_news.txt",
        tokenizer,
        max_length=max_length,
        cache_path="model/tokenized_fullcorpus.pt",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GeneratorTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_length=max_length,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        model.train()
        epoch_loss = 0.0

        start_time = time.time()
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")

        for batch_idx, (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            if (batch_idx + 1) % 1000 == 0:
                pbar.set_postfix(
                    {
                        "batch_loss": loss.item(),
                        "avg_loss": epoch_loss / (batch_idx + 1),
                    }
                )
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1} completed. Mean loss: {avg_loss:.4f}. Time: {elapsed/60:.2f} min."
        )
        os.makedirs("model", exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict()}, f"model/checkpoint_fullcorpus.pt"
        )


if __name__ == "__main__":
    train()
