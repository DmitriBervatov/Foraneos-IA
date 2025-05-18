# train.py
import torch, pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model.gpt import GPT

SEQ_LEN     = 128
BATCH_SIZE  = 32
EPOCHS      = 5
LR          = 1e-4
VOCAB_SIZE  = 25_000            # igual a tu vocab.json

class TokenDataset(Dataset):
    def __init__(self, token_ids):
        self.ids = token_ids
    def __len__(self):
        return len(self.ids) - SEQ_LEN
    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx+SEQ_LEN],     dtype=torch.long)
        y = torch.tensor(self.ids[idx+1:idx+SEQ_LEN+1], dtype=torch.long)
        return x, y

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🔌 Dispositivo:", device)

    # cargar ids
    with open("data/train_data.pkl", "rb") as f:
        ids = pickle.load(f)
    print(f"🗄  Tokens totales: {len(ids):,}")

    loader = DataLoader(TokenDataset(ids), batch_size=BATCH_SIZE, shuffle=True)
    model  = GPT(VOCAB_SIZE, SEQ_LEN).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn= nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        loop, total = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"), 0
        model.train()
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} finalizada. Loss medio: {total/len(loader):.4f}")

    torch.save(model.state_dict(), "model/gpt_model.pth")
    print("💾 Modelo guardado en model/gpt_model.pth")

if __name__ == "__main__":
    main()
