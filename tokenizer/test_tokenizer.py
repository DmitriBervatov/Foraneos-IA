from tokenizer.tokenizer import BPETokenizer

if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.load_vocab("tokenizer/vocab.json")

    texto_prueba = "Bone Broth is delicious and healthy"
    tokens = tokenizer.tokenize(texto_prueba)
    print("Tokens:", tokens)

    token_ids = tokenizer.tokens_to_ids(tokens)
    print("Token IDs:", token_ids)
