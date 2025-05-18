# model/gpt.py
import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len,
                 d_model=512, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True         # ✅ corrige la advertencia
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        self.ln_f   = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size)

    def forward(self, x):                   # x: (B, T)
        tok = self.token_emb(x)             # (B, T, C)
        pos = self.pos_emb[:, :x.size(1), :]
        h   = tok + pos
        h   = self.transformer(h)           # (B, T, C)
        h   = self.ln_f(h)
        return self.head(h)                 # (B, T, vocab_size)
