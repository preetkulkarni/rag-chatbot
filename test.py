"""
# testing ollama

import subprocess

prompt = "You are a helpful assistant. What's the capital of France?"

response = subprocess.run(
    ["ollama", "run", "llama3.1"],
    input=prompt,
    capture_output=True,
    text=True
)

print("Response from LLaMA 3.1:\n")
print(response.stdout.strip())
"""

"""
# testing text extraction from pdf

from modules.loader import extract_text_from_pdf

pdf_text = extract_text_from_pdf()
print("\n--- Extracted PDF Text ---\n")
print(pdf_text[:1000]) 
"""
"""
# testing chunking
from modules.loader import extract_text_from_pdf
from modules.chunker import split_text_into_chunks

text = extract_text_from_pdf()

chunks = split_text_into_chunks(text)

print("\n Total Chunks: {len(chunks)}\n")
print("Sample Chunk: \n")
print(chunks[300])
"""

# testing top k chunk retrieval
# make sure .pkl and .index files present
from modules.retriever import retrieve_top_k_chunks

query = "Is knee surgery covered in a 3-month old policy?"
top_chunks = retrieve_top_k_chunks(query)

print("\nTop Matching Chunks: \n")
for i, chunk in enumerate(top_chunks):
    print(f"[{i+1}]\n{chunk}\n")






