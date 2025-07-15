import numpy as np
from modules.embedder import embed_model, load_index, load_chunks
from modules import config

def retrieve_top_k_chunks(query, k=config.TOP_K, index_path="faiss.index", chunk_path="chunks.pkl"):
    # returns array of top K relevant chunks

    index = load_index(index_path)
    chunks = load_chunks(chunk_path)

    query_embedding = embed_model.encode([query])

    distances, indices = index.search(np.array(query_embedding), k)

    top_chunks = [chunks[i] for i in indices[0]]
    return top_chunks
