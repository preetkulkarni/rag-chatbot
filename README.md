
# AI-Powered Insurance Policy Analyst

A sophisticated RAG chatbot that acts as an intelligent claims assistant, providing clear, evidence-based coverage verdicts from any insurance policy document.



## Description

This project leverages a Retrieval-Augmented Generation (RAG) pipeline to analyze complex insurance policies. Upload a policy, ask a question about a claim scenario, and receive a confident, verifiable verdict—Approved, Not Approved, or Insufficient Information—based strictly on the text within the document.




## The RAG Pipeline

- **Load & Clean**: Ingests a PDF, cleans the text, and normalizes its structure.

- **Chunk**: Breaks the document into small, semantically meaningful pieces of text.

- **Embed**: Converts each chunk into a numerical vector using `bge-large-en-v1.5`.

- **Index**: Stores these vectors in a high-speed FAISS index for fast searching.

- **Retrieve & Re-rank**: When a query is asked, it retrieves the top 15 potential chunks and uses `bge-reranker-large` to select the absolute best 3.

- **Generate**: The top chunks and the query are sent to the LLM with a strict, rule-based prompt to generate the final verdict and justification.
## Features

- 📄 **Dynamic Policy Upload**: Analyze any insurance policy PDF on the fly. The system creates a persistent, cached index for each document, allowing for instant analysis in future sessions.

- 🧹 **Intelligent Preprocessing**: A sophisticated pipeline cleans document text, intelligently removes repetitive headers/footers, and normalizes legal and medical terminology for maximum accuracy.

- 🎯 **High-Precision Retrieval Engine**:

   **Deep Semantic Search**: Uses the powerful `BAAI/bge-large-en-v1.5 model` to understand the nuances of complex insurance clauses.

   **Re-ranking**: A fast vector search retrieves initial candidates, which are then meticulously re-ranked by `BAAI/bge-reranker-large` to find the most precise evidence for your query.

- ⚖️ **Strict, Rule-Based Verdicts**: The chatbot is engineered to act as a strict analyst, never inferring or assuming information. It delivers one of three unambiguous verdicts based only on explicit text.

- 🤝 **Interactive Clarification**: If a query is ambiguous, the bot doesn't just fail—it asks targeted, clarifying questions to gather the necessary details, turning a simple Q&A into a collaborative analysis session.

- 🔍 **Verifiable Evidence Trail**: Every verdict is backed by sources, citing the exact page number from the policy document, ensuring complete transparency and trust.


## Tech Stack

**Language**: Python

**Core ML/NLP**: PyTorch, Sentence-Transformers, LlamaIndex

**Vector Search**: FAISS (CPU)

**PDF Processing**: PyMuPDF (fitz)

**Local LLM Server**: Ollama

## Run Locally

**Prerequisites**
- Python 3.9 or higher.

- An active instance of Ollama running in the background.

- A downloaded Ollama model (e.g., llama3.1). You can get this by running:

```bash
ollama run llama3.1
```

**Installation & Usage**
- Clone the repository:

```bash
git clone https://github.com/preetkulkarni/rag-chatbot.git
```
```bash
cd rag-chatbot
```
- Install dependencies:
```bash
pip install -r requirements.txt
```
- Run the application:
```bash
python main.py
```
**Follow the on-screen prompts to provide a path to your PDF policy document and start asking questions!**

⚠️ **Important Warnings**
- **High Memory Usage**: This application loads multiple large AI models and requires significant system RAM (8 GB or more is recommended). Please close other memory-intensive programs (like web browsers) before running.

- **Ollama is Required**: This is not a standalone application. It will not function unless you have a local instance of Ollama installed and running.

## 🔗 Links

Thanks for checking out my project!

Let's connect on LinkedIn:

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/preet-kulkarni-2453ab284/)

