import os
import shutil
from modules.loader import extract_and_clean_pdf
from modules.chunker import get_text_nodes
from modules.embedder import embed_chunks, create_faiss_index
from modules.retriever import retrieve_top_k_chunks 
from modules.llm_interface import query_llm_with_context
from modules.persistence import save_data, cache_exists, get_cache_path

def build_index(file_path: str):
    print(f"\n🔄 Building index for {os.path.basename(file_path)}...")
    cache_path = get_cache_path(file_path)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    processed_data = extract_and_clean_pdf(file_path)
    if not processed_data or not processed_data.get("pages"):
        return None
    nodes = get_text_nodes(processed_data, file_path)
    if not nodes:
        return None
    embeddings = embed_chunks(nodes)
    index = create_faiss_index(embeddings)
    save_data(cache_path, "faiss.index", index, serializer='faiss')
    save_data(cache_path, "chunks.pkl", nodes, serializer='pickle')
    print(f"✅ Index for {os.path.basename(file_path)} built and saved.\n")
    return cache_path

def chat_session(file_path: str, cache_path: str):
    print(f"\n--- Chatting with: {os.path.basename(file_path)} ---")
    print("> Type 'exit' to quit, 'rebuild' to re-index, or 'back' to choose another file.\n")

    original_query = None
    
    while True:
        # new conversation
        if not original_query:
            user_input = input("Ask a question: ").strip()
            if not user_input:
                continue
            original_query = user_input
        else:
            pass

        if original_query.lower() in ["exit", "back", "rebuild"]:
            command = original_query.lower()
            original_query = None # Reset for the next loop
            if command == "exit": raise SystemExit("Goodbye!")
            if command == "back":
                print("\n--- Returning to file selection menu ---")
                break
            if command == "rebuild":
                build_index(file_path)
                continue
        
        query_to_send = original_query
        
        try:
            top_nodes = retrieve_top_k_chunks(query_to_send, cache_path)
            if not top_nodes:
                print("Could not retrieve relevant context. Please try another question.")
                original_query = None # Reset for next question
                continue

            llm_response = query_llm_with_context(query_to_send, top_nodes)
            
            print("\n📄 Answer:\n")
            print(llm_response.get("answer"))

            print("\nSources:")
            for node in top_nodes:
                metadata = node.metadata
                print(f"- {metadata['file_name']} (Page: {metadata['page_number']}, Chunk: {metadata['chunk_number']})")
            print("\n" + "=" * 50 + "\n")

            if llm_response.get("status") == "insufficient":
                print("The chatbot needs more information to provide a final answer:")
                for q in llm_response.get("questions", []):
                    print(f"- {q}")
                
                additional_info = input("\nPlease provide the requested details (or type 'skip' to ask a new question): ").strip()

                if additional_info.lower() == 'skip':
                    original_query = None # Reset to ask a new question
                    print("\n" + "="*50 + "\n")
                    continue
                
                # new query for next iteration
                original_query = f"{original_query} [User has provided new information: {additional_info}]"
                print("\n--- Thank you. Re-evaluating with new information... ---")
            else:
                original_query = None

        except Exception as e:
            print(f"❌ An error occurred: {e}")
            original_query = None 

def main():
    print("\n=== Dynamic RAG Chatbot (v3.3) ===")
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
                continue
            cache_path = new_cache_path
        else:
            print(f"✅ Found existing index for '{os.path.basename(file_path_input)}'.")
        try:
            chat_session(file_path_input, cache_path)
        except SystemExit as e:
            print(e)
            break

if __name__ == '__main__':
    main()
