
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Add it to your .env file."
    )

if not MODEL_NAME:
    raise ValueError(
        "GEMINI_MODEL is missing. Add an available model "
        "name to your .env file."
    )

client = genai.Client(api_key=API_KEY)


def generate_response(prompt: str) -> str:
    """Generate a response using the configured Gemini model."""

    result = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": 0.2,
            "max_output_tokens": 400,
        },
    )

    text = result.text

    if not text or not text.strip():
        raise RuntimeError("Gemini returned an empty response.")

    return text.strip()
