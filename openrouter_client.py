import base64
from pathlib import Path

import requests

import config


class OpenRouterError(RuntimeError):
    pass


def _headers(api_key: str) -> dict:
    if not api_key:
        raise OpenRouterError(
            "No OpenRouter API key set. Add OPENROUTER_API_KEY to your .env "
            "or paste one in the UI."
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/pdf-rag",
        "X-Title": "PDF RAG",
    }


def chat_completion(
    messages: list[dict],
    model: str,
    api_key: str,
    temperature: float = 0.2,
    max_tokens: int = 900,
    top_p: float | None = None,
    top_k: int | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k

    resp = requests.post(
        f"{config.OPENROUTER_BASE_URL}/chat/completions",
        headers=_headers(api_key),
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        raise OpenRouterError(f"OpenRouter error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        raise OpenRouterError(f"Unexpected OpenRouter response: {data}") from e

    content = message.get("content")
    if content is None:
        # Reasoning ("thinking") models can burn the whole max_tokens budget
        # on their hidden reasoning field and get cut off before writing an
        # actual answer.
        if message.get("reasoning"):
            raise OpenRouterError(
                "Model spent its whole token budget thinking and never wrote "
                "an answer - increase max_tokens (config.TEXT_MAX_TOKENS) "
                "and try again."
            )
        raise OpenRouterError(f"Unexpected OpenRouter response: {data}")
    return content.strip()


def caption_image(image_path: Path, model: str, api_key: str, context_hint: str = "") -> str:
    image_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    ext = Path(image_path).suffix.lstrip(".") or "png"
    data_url = f"data:image/{ext};base64,{b64}"

    prompt = (
        "Describe this figure/chart/image extracted from a financial PDF "
        "report. Mention the figure type (bar chart, table screenshot, "
        "logo, diagram, etc.), any titles, axis labels, categories, and "
        "the approximate values or trend shown. Be concise (3-5 sentences) "
        "and factual - only describe what's visible."
    )
    if context_hint:
        prompt += f"\n\nSurrounding page text for context:\n{context_hint[:500]}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]
    return chat_completion(messages, model=model, api_key=api_key, max_tokens=300)
