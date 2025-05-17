import pandas as pd

def load_data():
    recetas = pd.read_csv('data/recetas.csv')
    alimentos = pd.read_csv('data/alimentos.csv')
    ejercicios = pd.read_csv('data/ejercicios.csv')
    return recetas, alimentos, ejercicios