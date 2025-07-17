import faiss
import numpy as np
from typing import List
from llama_index.core.schema import TextNode
from . import config
from .persistence import load_data
from .embedder import model

def retrieve_top_k_chunks(query: str) -> List[TextNode]:
    """
    Retrieves the top-k most relevant TextNode objects for a given query.
    """
    print(f"\n🔍 Retrieving context for query...\n")
    
    try:
        index = load_data("faiss.index", serializer='faiss')
        nodes = load_data("chunks.pkl", serializer='pickle')
    except FileNotFoundError:
        print("❌ Error: Index or chunks file not found. Please build the index first.")
        return None

    query_embedding = model.encode([query]).astype('float32')

    distances, indices = index.search(query_embedding, config.TOP_K)

    retrieved_nodes = [nodes[i] for i in indices[0]]
    
    return retrieved_nodes
