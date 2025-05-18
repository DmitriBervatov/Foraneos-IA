from tokenizer.tokenizer import BPETokenizer

def cargar_texto(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return f.readlines()

if __name__ == "__main__":
    texto = cargar_texto("data/dataset.txt")
    tokenizer = BPETokenizer(vocab_size=25000)
    tokenizer.train(texto)
    tokenizer.save_vocab()
    print("✅ Tokenizador entrenado y guardado en vocab.json")
