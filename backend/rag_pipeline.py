from groq import Groq
from retriever import search
import os
from config import GROQ_API_KEY


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def generate_answer(query: str, top_k: int = 3):
    docs = search(query, top_k)
    context = "\n\n".join([f"Q: {d['question']}\nA: {d['answer']}" for d in docs])

    prompt = f"""
    You are a helpful medical assistant.
    Use the following context to answer the question.

    Context:
    {context}

    Question: {query}
    Answer:
    """
    '''client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)'''
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
        {
            "role": "system",
            "content": prompt,
        }
    ],
    )

    return {
        "query": query,
        "answer": response.choices[0].message.content.strip(),
        
    }