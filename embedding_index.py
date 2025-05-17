from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SemanticSearch:
    def __init__(self, texts, metadata):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.texts = texts
        self.metadata = metadata
        self.embeddings = self.model.encode(texts, convert_to_tensor=True)
        
    def query(self, question: str):
        question_embedding = self.model.encode([question], convert_to_tensor=True)
        scores = cosine_similarity(question_embedding, self.embeddings)[0]
        best_idx = np.argmax(scores)
        return self.metadata[best_idx], float(scores[best_idx])