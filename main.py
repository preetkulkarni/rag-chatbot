from modules.loader import extract_and_clean_pdf
from modules.chunker import split_text_into_chunks
from modules.embedder import embed_chunks, create_faiss_index
from modules.retriever import retrieve_top_k_chunks
from modules.llm_interface import query_llm_with_context
from modules.persistence import save_data, cache_exists

def build_index():
    """Builds and saves the index and chunks from the source PDF."""
    print("🔄 Building index from PDF...")
    text = extract_and_clean_pdf()
    chunks = split_text_into_chunks(text)
    embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)

    save_data("faiss.index", index, serializer='faiss')
    save_data("chunks.pkl", chunks, serializer='pickle')

    print("✅ Index and chunks saved.\n")

def main():
    print("\n=== RAG Chatbot for Policy Documents ===\n")
    print("> type 'exit' to quit")
    print("> type 'rebuild' to rebuild index and chunks\n")
    print("Current status: ")
    
    if not cache_exists(["faiss.index", "chunks.pkl"]):
        build_index()
    else:
        print("✅ FAISS index and chunks already exist.\n")

    while True:
        user_query = input("Ask a question: ").strip()

        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        elif user_query.lower() == "rebuild":
            build_index()
            continue

        try:
            top_chunks = retrieve_top_k_chunks(user_query)
            if top_chunks is None:
                print("Could not retrieve context. Please try rebuilding the index.")
                continue
            
            answer = query_llm_with_context(user_query, top_chunks)
            print("\n📄 Answer:\n")
            print(answer)
            print("\n" + "=" * 40 + "\n")
        
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again or rebuild the index if needed.\n")

if __name__ == '__main__':
    main()