from sentence_transformers import SentenceTransformer
from modules import config
import faiss

embed_model = SentenceTransformer(config.EMBED_MODEL)

def embed_chunks(chunks):
    return embed_model.encode(chunks, show_progress_bar=True)

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index