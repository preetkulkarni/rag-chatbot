from modules.loader import extract_and_clean_pdf
from modules.chunker import get_text_nodes
from modules.embedder import embed_chunks, create_faiss_index
from modules.retriever import retrieve_top_k_chunks
from modules.llm_interface import query_llm_with_context
from modules.persistence import save_data, cache_exists
from modules import config

def build_index():
    """Builds and saves the index and structured TextNodes from the source PDF."""
    print("\n🔄 Building index from PDF...")

    text = extract_and_clean_pdf(config.PDF_PATH)
    if not text:
        print("❌ Failed to extract text. Aborting index build.")
        return

    nodes = get_text_nodes(text, config.PDF_PATH)
    if not nodes:
        print("❌ Failed to create text nodes. Aborting index build.")
        return

    embeddings = embed_chunks(nodes)

    index = create_faiss_index(embeddings)
    save_data("faiss.index", index, serializer='faiss')

    save_data("chunks.pkl", nodes, serializer='pickle')

    print("✅ Index and text nodes saved.\n")

def main():
    print("\n=== RAG Chatbot for Policy Documents (v2.1) ===\n")
    print("> type 'exit' to quit")
    print("> type 'rebuild' to rebuild index and chunks\n")
    print("Current status: ")
    
    if not cache_exists(["faiss.index", "chunks.pkl"]):
        build_index()
    else:
        print("✅ FAISS index and text nodes already exist.\n")

    while True:
        user_query = input("Ask a question: ").strip()

        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        elif user_query.lower() == "rebuild":
            build_index()
            continue

        try:
            top_nodes = retrieve_top_k_chunks(user_query)
            if top_nodes is None:
                print("Could not retrieve context. Please try rebuilding the index.")
                continue
            
            answer = query_llm_with_context(user_query, top_nodes)
            print("\n📄 Answer:\n")
            print(answer)
            print("\n" + "=" * 40 + "\n")
        
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again or rebuild the index if needed.\n")

if __name__ == '__main__':
    main()