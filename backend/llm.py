import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _get_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def ask_llm(prompt: str) -> str:
    # Get relevant context from uploaded documents
    from .rag_engine import query_rag

    context = query_rag(prompt)

    system_prompt = """You are a helpful Kerala Technological University academic assistant. 
    Use the provided context from the syllabus and notes to answer questions accurately. 
    If the context doesn't contain relevant information, say so and provide general guidance."""

    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"

    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content or ""
