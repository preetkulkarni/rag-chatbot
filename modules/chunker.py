import os
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from typing import List
from modules import config

# v1 functions: split_text_into_chunks(text) 

def get_text_nodes(text: str, file_name: str) -> List[TextNode]:
    # splits long text into smaller TextNode objects

    if not text.strip():
        print("⚠️ Warning: Input text is empty. Returning no nodes.")
        return []
    
    print(f"Chunking text from '{file_name}'...")

    splitter = SentenceSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP,
    )

    string_chunks = splitter.split_text(text)

    nodes = []
    nodes = []
    for i, chunk_text in enumerate(string_chunks):
        node = TextNode(
            text=chunk_text,
            metadata={
                "file_name": os.path.basename(file_name), 
                "chunk_number": i + 1,
                # more metadata to be added here
            }
        )
        nodes.append(node)
    
    print(f"✅ Successfully created {len(nodes)} text nodes.")
    return nodes