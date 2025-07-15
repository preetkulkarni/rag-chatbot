from modules.loader import extract_text_from_pdf
from modules.chunker import split_text_into_chunks
from modules.embedder import embed_chunks, create_faiss_index, save_index, save_chunks, load_chunks
from modules.retriever import retrieve_top_k_chunks
from modules.llm_interface import query_llm_with_context

import os

def build_index_if_needed():
    if os.path.exists("faiss.index") and os.path.exists("chunks.pkl"):
        print("✅ FAISS index and chunks already exist.")
        return

    print("🔄 Building index from PDF...")
    text = extract_text_from_pdf()
    chunks = split_text_into_chunks(text)
    embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)
    save_index(index)
    save_chunks(chunks)
    print("✅ Index and chunks saved.")

def main():
    print("\n=== RAG Chatbot for Policy Documents ===\n")
    build_index_if_needed()

    while True:
        user_query = input("Ask a question (or type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            break

        top_chunks = retrieve_top_k_chunks(user_query)
        answer = query_llm_with_context(user_query, top_chunks)

        print("\n📄 Answer:\n")
        print(answer)
        print("\n" + "=" * 40 + "\n")


if __name__ == '__main__':
    main()
