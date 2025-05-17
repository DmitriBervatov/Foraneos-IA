from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_data
from embedding_index import SemanticSearch

app = FastAPI()

recetas, alimentos, ejercicios = load_data()

knowledge = recetas["descripcion"].tolist() + alimentos["descripcion"].tolist() + ejercicios["descripcion"].tolist()
metadata = recetas["nombre"].tolist() + alimentos["nombre"].tolist() + ejercicios["nombre"].tolist()

semantic_search = SemanticSearch(knowledge, metadata)

class Question(BaseModel):
    pregunta: str
    
@app.post("/ask")
def ask_agent(q: Question):
    respuesta, score = semantic_search.query(q.pregunta)
    return {
        "respuesta": respuesta,
        "score": score
    }