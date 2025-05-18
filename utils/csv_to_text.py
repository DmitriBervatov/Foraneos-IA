import pandas as pd
import glob
import os


def csvs_a_texto(carpeta="data", archivo_salida="data/dataset.txt"):
    textos = []
    archivos = glob.glob(os.path.join(carpeta, "*.csv"))

    for ruta in archivos:
        df = pd.read_csv(ruta)
        for _, fila in df.iterrows():
            linea = " | ".join([f"{col}: {str(fila[col])}" for col in df.columns])
            textos.append(linea)

    with open(archivo_salida, "w", encoding="utf-8") as f:
        for t in textos:
            f.write(t + "\n")


if __name__ == "__main__":
    csvs_a_texto()
