import os
import shutil
from modules.loader import extract_and_clean_pdf
from modules.chunker import get_text_nodes
from modules.embedder import embed_chunks, create_faiss_index
from modules.retriever import retrieve_top_k_chunks 
from modules.llm_interface import query_llm_with_context
from modules.persistence import save_data, cache_exists, get_cache_path

def build_index(file_path: str):

    print(f"🔄 Building index for {os.path.basename(file_path)}...")
    
    cache_path = get_cache_path(file_path)
    
    if os.path.exists(cache_path):
        print(f"🗑️ Clearing old cache at {cache_path}...")
        shutil.rmtree(cache_path)

    processed_data = extract_and_clean_pdf(file_path)
    if not processed_data or not processed_data.get("pages"):
        print("❌ Failed to extract text. Aborting index build.")
        return None

    nodes = get_text_nodes(processed_data, file_path)
    if not nodes:
        print("❌ Failed to create text nodes. Aborting index build.")
        return None

    embeddings = embed_chunks(nodes)
    index = create_faiss_index(embeddings)
    
    save_data(cache_path, "faiss.index", index, serializer='faiss')
    save_data(cache_path, "chunks.pkl", nodes, serializer='pickle')

    print(f"✅ Index for {os.path.basename(file_path)} built and saved.\n")
    return cache_path

def chat_session(file_path: str, cache_path: str):
    # chat session for a specific doc

    print(f"\n--- Chatting with: {os.path.basename(file_path)} ---")
    print("> type 'exit' to quit, 'rebuild' to re-index this file, or 'back' to choose another file.\n")
    
    while True:
        user_query = input("Ask a question: ").strip()

        if user_query.lower() == "exit":
            print("Goodbye!")
            raise SystemExit() # might be problematic
        
        if user_query.lower() == "back":
            print("\n--- Returning to file selection menu ---")
            break

        if user_query.lower() == "rebuild":
            build_index(file_path)
            continue

        try:
            top_nodes = retrieve_top_k_chunks(user_query, cache_path)
            if not top_nodes:
                print("Could not retrieve relevant context. Please try another question or 'rebuild'.")
                continue

            answer = query_llm_with_context(user_query, top_nodes)
            
            print("\n📄 Answer:\n")
            print(answer)

            print("\nSources:")
            sources = [node.metadata for node in top_nodes]
            for src in sources:
                print(f"- {src['file_name']} (Page: {src['page_number']}, Chunk: {src['chunk_number']})")

            print("\n" + "=" * 50 + "\n")
        
        except Exception as e:
            print(f"❌ An error occurred: {e}")

def main():
    print("\n=== Dynamic RAG Chatbot (v2.3) ===")
    
    while True:
        file_path_input = input("Enter the path to a PDF file (or type 'exit' to quit): ").strip()

        if file_path_input.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.exists(file_path_input) or not file_path_input.lower().endswith('.pdf'):
            print("❌ Invalid path. Please provide a valid path to a PDF file.")
            continue
        
        cache_path = get_cache_path(file_path_input)

        if not cache_exists(cache_path, ["faiss.index", "chunks.pkl"]):
            print(f"File '{os.path.basename(file_path_input)}' has not been indexed yet.")
            new_cache_path = build_index(file_path_input)
            if not new_cache_path:
                print(f"❌ Cache build failed. Exiting to main menu...")
                continue
            cache_path = new_cache_path
        else:
            print(f"✅ Found existing index for '{os.path.basename(file_path_input)}'.")

        # start a chat session
        try:
            chat_session(file_path_input, cache_path)
        except SystemExit as e: # catch the exit command
            print(e)
            break

if __name__ == '__main__':
    main()
