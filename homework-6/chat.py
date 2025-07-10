import torch
from tokenizers import Tokenizer

from generator_transformer import GeneratorTransformer


def chat():
    tokenizer = Tokenizer.from_file("tokenizer/russian_tokenizer.json")
    model = GeneratorTransformer.load_from_checkpoint(
        "model/checkpoint_fullcorpus.pt",
        vocab_size=tokenizer.get_vocab_size(),
        d_model=128,
        nhead=4,
        num_layers=2,
        max_length=128,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    eos_token_id = tokenizer.token_to_id("<eos>")

    while True:
        user_input = input("Вы: ")
        if user_input.lower() == "quit":
            break

        mode = input("Режим (1 - обычный, 2 - beam search): ")
        if mode == "2":
            response = model.beam_search(
                tokenizer,
                user_input,
                context_len=50,
                max_out_tokens=50,
                beam_width=3,
                eos_token_id=eos_token_id,
            )
        else:
            response = model.generate(
                tokenizer,
                user_input,
                context_len=50,
                temperature=0.8,
                max_out_tokens=50,
                eos_token_id=eos_token_id,
            )
        print(f"Бот: {response}")


if __name__ == "__main__":
    chat()
