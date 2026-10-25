"""OpenRouter client. The API key never leaves the server (use Replit Secrets)."""

import json
import os
import re

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

TEXT_MODEL = os.getenv("TEXT_MODEL", "moonshotai/kimi-k2-0905")
# Any vision-capable model on OpenRouter. Check openrouter.ai/models and set VISION_MODEL in Secrets.
VISION_MODEL = os.getenv("VISION_MODEL", "google/gemini-2.5-flash")


def _headers() -> dict:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (add it in Replit Secrets).")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-Title": "MedGuide",
    }


def _extract_json(text: str) -> dict:
    """Models sometimes wrap JSON in ```; strip that and parse."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def chat_json(messages: list[dict], model: str, max_tokens: int = 900, timeout: float = 60.0) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,  # low: we want faithful rewriting, not creativity
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        # Ask OpenRouter to route only to providers that do not store/train on prompts.
        "provider": {"data_collection": "deny"},
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(OPENROUTER_URL, headers=_headers(), json=body)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return _extract_json(content)


def identify_from_image(image_b64: str, mime: str, prompt: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            ],
        }
    ]
    return chat_json(messages, VISION_MODEL, max_tokens=300)
