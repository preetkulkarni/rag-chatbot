import faiss
import numpy as np
from typing import List
from llama_index.core.schema import TextNode
from sentence_transformers import CrossEncoder
from . import config
from .persistence import load_data
from .embedder import model as embedding_model

reranker_model = CrossEncoder(config.RERANKER_MODEL)

def retrieve_top_k_chunks(query: str) -> List[TextNode]:
    # retrives top K most relevant chunks

    print(f"\n🔍 Retrieving context for query...\n")
    
    try:
        index = load_data("faiss.index", serializer='faiss')
        nodes = load_data("chunks.pkl", serializer='pickle')
    except FileNotFoundError:
        print("❌ Error: Index or chunks file not found. Please build the index first.")
        return None

    # retrieval of first set of chunks
    print(f"🔄 Stage 1: Retrieving top {config.TOP_K_INITIAL} initial candidates from vector store...")
    
    query_embedding = embedding_model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, config.TOP_K_INITIAL)
    initial_candidates = [nodes[i] for i in indices[0]]

    # re rank initial chunks
    print(f"🔄 Stage 2: Re-ranking the {len(initial_candidates)} candidates for higher precision...")

    rerank_pairs = [[query, node.get_content()] for node in initial_candidates]
    
    # attaching relevance score to chunks
    scores = reranker_model.predict(rerank_pairs)
    scored_candidates = list(zip(scores, initial_candidates))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # final chunks
    final_top_k = config.TOP_K_FINAL
    final_nodes = [candidate for score, candidate in scored_candidates[:final_top_k]]
    
    print(f"✅ Re-ranking complete. Selected top {len(final_nodes)} nodes.\n")
    return final_nodes
