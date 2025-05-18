import torch
from model.model import GPTSimple
from tokenizer.tokenizer import BPETokenizer
import json
import random


def top_k_top_p_filtering(logits, top_k=50, top_p=0.9, temperature=1.0):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    if top_p < 1.0:
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    return logits


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.tokenize(prompt)
    print("input_ids:", input_ids)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_tensor)
            print("logits shape:", logits.shape)
            next_token_logits = logits[0, -1, :]
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p,
                                                    temperature=temperature)
            probabilities = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

    generated_ids = input_tensor[0].tolist()
    return tokenizer.detokenize(generated_ids)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    with open("tokenizer/vocab.json", "r") as f:
        vocab = json.load(f)

    tokenizer = BPETokenizer(vocab=vocab)

    model = GPTSimple(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load("model/gpt_simple.pth", map_location=device))

    prompt = input("Escribe el prompt inicial: ")
    if prompt.strip():
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=100,
            temperature=1.0,
            top_k=20,
            top_p=0.85
        )

        print("\n📝 Texto generado:")
        print(generated)
        print("-" * 80)
    else:
        print("No se ingresó prompt. Terminando programa.")


if __name__ == "__main__":
    main()
