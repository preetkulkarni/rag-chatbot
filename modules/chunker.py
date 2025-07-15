from llama_index.core.node_parser.text import SentenceSplitter
from modules import config

def split_text_into_chunks(text):
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks = splitter.split_text(text)
    return chunks
