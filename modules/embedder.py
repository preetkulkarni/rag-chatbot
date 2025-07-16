from sentence_transformers import SentenceTransformer
from modules import config
import numpy as np
import faiss
import os
import pickle

embed_model = SentenceTransformer(config.EMBED_MODEL)

def embed_chunks(chunks):
    # returns 2d array of embeddings

    return embed_model.encode(chunks, show_progress_bar=True)

def create_faiss_index(embeddings):
    # returns faiss.IndexFlatL2: FAISS index for similar searches
    # L2 = euclidean distance

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# directory for cached files (faiss.index & chunks.pkl)
dir = config.CACHED_DIR

def save_index(index, path="faiss.index"):
    full_path = os.path.join(dir, path)
    faiss.write_index(index, full_path)

def load_index(path="faiss.index"):
    full_path = os.path.join(dir, path)
    return faiss.read_index(full_path)

def save_chunks(chunks, path="chunks.pkl"):
    full_path = os.path.join(dir, path)
    with open(full_path, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks(path="chunks.pkl"):
    full_path = os.path.join(dir, path)
    with open(full_path, "rb") as f:
        return pickle.load(f)