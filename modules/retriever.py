import numpy as np
from modules.embedder import embed_model
from modules.persistence import load_data # Updated import
from modules import config

def retrieve_top_k_chunks(query, k=config.TOP_K):
    # returns top K relevant chunks
    
    index = load_data("faiss.index", serializer='faiss')
    chunks = load_data("chunks.pkl", serializer='pickle')
    
    if index is None or chunks is None:
        print("⚠️ Cache is missing or corrupted. Please use the 'rebuild' command.")
        return None

    query_embedding = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    
    top_chunks = [chunks[i] for i in indices[0]]
    return top_chunks