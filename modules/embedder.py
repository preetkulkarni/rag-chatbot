from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List
from llama_index.core.schema import TextNode
from modules import config

model = SentenceTransformer(config.EMBED_MODEL)

def embed_chunks(nodes: List[TextNode]) -> np.ndarray:
    if not nodes:
        print("⚠️ Warning: No nodes to embed. Returning empty array.")
        return np.array([])
    
    texts_to_embed = [node.get_content() for node in nodes]
    print("\nGenerating embeddings...")
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    faiss.normalize_L2(embeddings)
    print("✅ Embeddings generated successfully.\n")
    return embeddings.astype('float32')

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:

    if embeddings.size == 0:
        raise ValueError("Cannot create FAISS index from empty embeddings array.")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index