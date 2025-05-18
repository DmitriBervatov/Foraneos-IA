# generate.py
import torch, json
from model.gpt import GPT
from tokenizer.tokenizer import BPETokenizer

SEQ_LEN    = 128
MAX_TOKENS = 80          # cuánto texto nuevo generar
TOP_K      = 40
TEMP       = 0.8

def top_k_sample(logits, k=TOP_K, temperature=TEMP):
    logits = logits / temperature
    topk   = torch.topk(logits, k)
    probs  = torch.softmax(topk.values, dim=-1)
    idx    = torch.multinomial(probs, 1)
    return topk.indices[idx]

def main():
    # cargar vocab & tokenizer
    with open("tokenizer/vocab.json") as f:
        vocab = json.load(f)
    tokenizer = BPETokenizer(vocab=vocab)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    model   = GPT(len(vocab), SEQ_LEN).to(device)
    model.load_state_dict(torch.load("model/gpt_model.pth", map_location=device))
    model.eval()

    prompt = input("📝 Prompt: ")
    ids    = tokenizer.tokenize(prompt)

    for _ in range(MAX_TOKENS):
        x = torch.tensor([ids[-SEQ_LEN:]], device=device)
        with torch.no_grad():
            logits = model(x)[0, -1]        # último paso
        next_id = top_k_sample(logits)
        ids.append(int(next_id))

    print("\n🖨️  Texto generado:\n")
    print(tokenizer.detokenize(ids))

if __name__ == "__main__":
    main()
