# clustering.py
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
import numpy as np

def cluster_snippets(snippets, model_name="all-MiniLM-L6-v2", n_clusters=None):
    texts = [s.get("snippet") or s.get("title") or "" for s in snippets]
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if n_clusters is None:
        n = max(2, int(np.sqrt(len(texts)/2 + 0.5)))
    else:
        n = n_clusters
    if len(texts) < n:
        n = max(1, len(texts))
    if n == 1:
        return [snippets]
    clustering = AgglomerativeClustering(n_clusters=n)
    labels = clustering.fit_predict(embs)
    clusters = {}
    for lbl, s in zip(labels, snippets):
        clusters.setdefault(int(lbl), []).append(s)
    clusters_list = sorted(clusters.values(), key=lambda c: -len(c))
    return clusters_list
