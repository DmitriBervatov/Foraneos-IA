import re
import json
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self, vocab=None, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.token_to_id = {}
        self.id_to_token = {}

        if vocab is not None:
            self.vocab = set(vocab)
            self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
            self.id_to_token = {i: tok for i, tok in enumerate(vocab)}

    def train(self, corpus):
        vocab = defaultdict(int)
        for line in corpus:
            for word in line.strip().split():
                word = tuple(word) + ("</w>",)
                vocab[word] += 1

        for i in range(self.vocab_size):
            print(f"[{i + 1}/{self.vocab_size}] Fusionando pares más frecuentes...")
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_vocab[tuple(new_word)] = freq
            vocab = new_vocab
            self.vocab.add("".join(best))

        # Crear mappings token -> id e id -> token
        self.token_to_id = {tok: i for i, tok in enumerate(sorted(self.vocab))}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    def tokenize(self, text):
        words = text.strip().split()
        tokens = []
        for word in words:
            chars = list(word) + ["</w>"]
            i = 0
            while i < len(chars):
                j = len(chars)
                found = None
                while j > i:
                    candidate = "".join(chars[i:j])
                    if candidate in self.vocab:
                        found = candidate
                        break
                    j -= 1
                if found:
                    tokens.append(found)
                    i = j
                else:
                    tokens.append(chars[i])
                    i += 1
        return self.tokens_to_ids(tokens)

    def detokenize(self, token_ids):
        tokens = [self.id_to_token[token_id] for token_id in token_ids if token_id in self.id_to_token]
        text = "".join(tokens)
        text = text.replace("</w>", " ").strip()
        return text

    def save_vocab(self, path="tokenizer/vocab.json"):
        with open(path, "w") as f:
            json.dump(sorted(list(self.vocab)), f, indent=2)

    def load_vocab(self, path="tokenizer/vocab.json"):
        with open(path, "r") as f:
            vocab_list = json.load(f)
        self.vocab = set(vocab_list)
        self.token_to_id = {tok: i for i, tok in enumerate(vocab_list)}
        self.id_to_token = {i: tok for i, tok in enumerate(vocab_list)}

    def tokens_to_ids(self, tokens):
        return [self.token_to_id[token] for token in tokens if token in self.token_to_id]
