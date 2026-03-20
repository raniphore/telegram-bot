"""
OpenAI API wrapper for answer generation.
"""
import os
from openai import AsyncOpenAI

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based ONLY on the "
    "provided context. If the answer is not in the context, say so clearly. "
    "Be concise and direct."
)

SUMMARY_SYSTEM_PROMPT = (
    "You are a helpful assistant. Summarize the following conversation history "
    "between a user and a bot. Be concise and highlight the key topics discussed."
)


async def generate_answer(query: str, context: str, history: list[dict] | None = None) -> str:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for turn in history:
            messages.append({"role": "user", "content": turn["content"]})
            messages.append({"role": "assistant", "content": turn["response"]})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}",
    })

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


async def generate_summary(history_text: str) -> str:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": history_text},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
