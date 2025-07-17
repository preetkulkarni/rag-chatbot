from typing import List
from llama_index.core.schema import TextNode
import subprocess

def build_prompt(user_query: str, context_nodes: List[TextNode]) -> str:

    if not context_nodes:
        return "I could not find any relevant information in the document to answer your question."

    context_text = "\n\n---\n\n".join([node.get_content() for node in context_nodes])
    prompt = f"""
You are a claims decision assistant helping users understand whether a given scenario is covered under an insurance policy.

Your job is to:
1. Interpret the user query, even if it is vague or incomplete.
2. Use ONLY the policy clauses in the context below to form your decision.
3. Give a clear, structured answer:
    - Justification: Detailed explanation with references to clause numbers.
    - Decision: One of the following verdicts, with a brief reason:
        • Approved — <brief reason>
        • Not Approved — <brief reason>
        • Insufficient Information — <brief reason>

--- START OF CONTEXT ---

{context_text}

--- END OF CONTEXT ---

User Query: "{user_query}"

Please respond using the following format exactly:

Justification:
<Your explanation here, citing relevant clauses.>

Decision: <Approved / Not Approved / Insufficient Information> — <brief one-line reason>

⚠️ Disclaimer: This decision is based solely on the policy clauses provided and is for informational purposes only. It does not constitute a legally binding or professional assessment.
"""
    return prompt.strip()



def query_llm_with_context(user_query, context_nodes, model_name="llama3.1"):
    # passes query with context chunks to LLM (llama3.1) and returns output 

    prompt = build_prompt(user_query, context_nodes)

    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"❌ LLM Error: {e.stderr or e.stdout}"
