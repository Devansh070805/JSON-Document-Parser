import google.generativeai as genai
from core.config import key

genai.configure(api_key=key.gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

def answer_question_with_context(question: str, context_chunks: list[str]) -> str:
    context = "\n".join(context_chunks)
    prompt = (
        f"You are a helpful assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini API Error: {str(e)}"
