from tokenizer.tokenizer import BPETokenizer
import pickle

def cargar_texto(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return f.read()

def preparar_datos(texto, tokenizer, seq_len=128):
    token_ids = tokenizer.tokenize(texto)

    # Dividir token_ids en secuencias de tamaño seq_len + 1 para input y target
    inputs = []
    targets = []
    for i in range(0, len(token_ids) - seq_len):
        input_seq = token_ids[i:i+seq_len]
        target_seq = token_ids[i+1:i+seq_len+1]
        inputs.append(input_seq)
        targets.append(target_seq)

    print(f"Preparados {len(inputs)} pares input-target de longitud {seq_len}")
    return inputs, targets

if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.load_vocab("tokenizer/vocab.json")

    texto = cargar_texto("data/dataset.txt")
    token_ids = tokenizer.tokenize(texto)

    print(f"✅ Tokenizado completo: {len(token_ids)} tokens")

    with open("data/train_data.pkl", "wb") as f:
        pickle.dump(token_ids, f)

    print("💾 Guardado como secuencia plana en data/train_data.pkl")
