# embeddings_store.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class EmbeddingStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dim)
        self.meta = []  # list of dicts (snippet,title,link)

    def add(self, texts, metas):
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embs = embs.astype('float32')
        self.index.add(embs)
        self.meta.extend(metas)

    def search(self, query, top_k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True).astype('float32')
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.meta):
                results.append(self.meta[idx])
        return results

    def all_meta(self):
        return self.meta
