import re
from typing import List, Dict, Any
from llama_index.core.schema import TextNode
import subprocess

def build_prompt(user_query: str, context_nodes: List[TextNode]) -> str:
    context_text = "\n\n---\n\n".join([node.get_content() for node in context_nodes])

    prompt = f"""
You are a strict, rule-based insurance analyst. You NEVER infer or make assumptions. Your decisions are based ONLY on explicit text found in the provided CONTEXT.

Your task is to follow these steps precisely:
1.  Analyze the user's query to understand the core question.
2.  Scrutinize the CONTEXT for clauses that EXPLICITLY approve or deny the user's request.
3.  Based on your analysis, make a final decision by following the rules below.

--- START OF CONTEXT ---
{context_text}
--- END OF CONTEXT ---

User Query: "{user_query}"

--- DECISION RULES ---
- **RULE A (Approval/Denial):** If the CONTEXT contains explicit text that confirms coverage or denial, your 'Decision' MUST be "Approved" or "Not Approved". In this case, the 'Clarifying Questions' section MUST be left completely blank.
- **RULE B (Insufficient Information):** If the CONTEXT is vague, does not mention the specific procedure, or requires ANY assumption on your part, your 'Decision' MUST be "Insufficient Information".
- **RULE C (Clarification):** IF AND ONLY IF your 'Decision' is "Insufficient Information", you MUST provide a numbered list of questions under 'Clarifying Questions' to resolve the ambiguity.

Respond using the following format EXACTLY:

Justification:
<Your detailed explanation here, citing ONLY explicit text from the context.>

Decision: <Approved / Not Approved / Insufficient Information> — <A brief, factual reason for your decision.>

Clarifying Questions:
<A numbered list of questions ONLY if the Decision is "Insufficient Information". This section MUST NOT BE SHOWN otherwise.>

---
Disclaimer: This decision is based solely on the policy clauses provided and is for informational purposes only.
"""
    return prompt.strip()

def query_llm_with_context(user_query: str, context_nodes: List[TextNode], model_name="llama3.1") -> Dict[str, Any]:
    prompt = build_prompt(user_query, context_nodes)

    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        raw_output = result.stdout.strip()
        
        response = {
            "status": "sufficient",
            "answer": raw_output,
            "questions": []
        }

        if "Decision: Insufficient Information" in raw_output:
            response["status"] = "insufficient"
            
            match = re.search(r"Clarifying Questions:\s*\n(.*?)(?=\n---|\Z)", raw_output, re.DOTALL)
            if match:
                questions_text = match.group(1).strip()
                questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
                response["questions"] = questions
                response["answer"] = raw_output.split("Clarifying Questions:")[0].strip()

        return response

    except subprocess.CalledProcessError as e:
        error_message = f"❌ LLM Error: {e.stderr or e.stdout}"
        return {"status": "error", "answer": error_message, "questions": []}
