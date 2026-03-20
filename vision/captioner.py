"""
Vision captioning using OpenAI GPT-4o (vision) API.
Generates a short caption and 3 keyword tags for uploaded images.
"""
import base64
import os
from openai import AsyncOpenAI

VISION_PROMPT = (
    "Describe this image in one concise sentence (caption), then list exactly "
    "3 relevant keyword tags. Format your response exactly as:\n"
    "Caption: <caption>\nTags: <tag1>, <tag2>, <tag3>"
)


async def describe_image(image_bytes: bytes) -> str:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=200,
    )

    text = response.choices[0].message.content.strip()
    return f"*Image Analysis*\n\n{text}"
