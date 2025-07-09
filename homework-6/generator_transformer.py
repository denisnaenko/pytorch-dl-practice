import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        max_length=192,
        pad_token_id=2,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_length, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(
            x.device
        )
        out = self.transformer(x, x, tgt_mask=tgt_mask)
        out = out.transpose(0, 1)  # (batch, seq, d_model)
        return self.fc(out)

    @staticmethod
    def load_from_checkpoint(path, **kwargs):
        checkpoint = torch.load(path, map_location="cpu")
        model = GeneratorTransformer(**kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def generate(
        self,
        tokenizer,
        prompt,
        context_len=50,
        temperature=1.0,
        max_out_tokens=50,
        eos_token_id=None,
    ):
        self.eval()
        device = next(self.parameters()).device
        input_ids = tokenizer.encode(prompt).ids
        input_ids = [tokenizer.token_to_id("<bos>")] + input_ids
        input_ids = torch.tensor([input_ids], device=device)
        generated = input_ids.clone()

        for _ in range(max_out_tokens):
            outputs = self(generated[:, -context_len:])
            next_token_logits = outputs[0, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return tokenizer.decode(generated[0].tolist())

    def beam_search(
        self,
        tokenizer,
        prompt,
        context_len=50,
        max_out_tokens=50,
        beam_width=3,
        eos_token_id=None,
    ):
        self.eval()
        device = next(self.parameters()).device

        input_ids = tokenizer.encode(prompt).ids
        input_ids = [tokenizer.token_to_id("<bos>")] + input_ids
        input_ids = torch.tensor([input_ids], device=device)

        sequences = [(input_ids, 0.0)]  # (tokens, score)

        for _ in range(max_out_tokens):
            all_candidates = []

            for seq, score in sequences:
                outputs = self(seq[:, -context_len:])

                logits = outputs[0, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

                for k in range(beam_width):
                    next_token = topk_indices[k].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + topk_log_probs[k].item()

                    all_candidates.append((new_seq, new_score))

            # prune
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered[:beam_width]

            # stop if all beams ended with EOS
            if eos_token_id is not None and all(
                seq[0][0, -1].item() == eos_token_id for seq in sequences
            ):
                break

        # return the best sequence
        best_seq = sequences[0][0][0].tolist()
        return tokenizer.decode(best_seq)
