from modules.loader import extract_and_clean_pdf
from modules.chunker import split_text_into_chunks
from modules.embedder import embed_chunks, create_faiss_index, save_index, save_chunks
from modules.retriever import retrieve_top_k_chunks
from modules.llm_interface import query_llm_with_context

import os

CACHE_DIR = "cached_files"
os.makedirs(CACHE_DIR, exist_ok=True)

def build_index_if_needed():
    if os.path.exists("cached_files/faiss.index") and os.path.exists("cached_files/chunks.pkl"):
        print("✅ FAISS index and chunks already exist.\n")
        return

    print("🔄 Building index from PDF...")
    text = extract_and_clean_pdf()
    chunks = split_text_into_chunks(text)
    embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)
    save_index(index)
    save_chunks(chunks)
    print("✅ Index and chunks saved.")

def rebuild():
    print("\n🔄 Rebuilding index from PDF...\n")
    text = extract_and_clean_pdf()
    chunks = split_text_into_chunks(text)
    embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)
    save_index(index)
    save_chunks(chunks)
    print("✅ Index and chunks saved.\n")
    
def main():
    print("\n=== RAG Chatbot for Policy Documents ===\n")
    print("> type 'exit' to quit")
    print("> type 'rebuild' to rebuild index and chunks\n")
    print("Current status: ")
    build_index_if_needed()

    while True:
        user_query = input("Ask a question: ").strip()

        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        elif user_query.lower() == "rebuild":
            rebuild()
            continue

        try:
            top_chunks = retrieve_top_k_chunks(user_query)
            answer = query_llm_with_context(user_query, top_chunks)

            print("\n📄 Answer:\n")
            print(answer)
            print("\n" + "=" * 40 + "\n")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or rebuild the index if needed.\n")


if __name__ == '__main__':
    main()
