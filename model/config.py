class GPTConfig:
    def __init__(self, vocab_size, block_size=64, n_layer=2, n_head=2, n_embd=128):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
