"""
Suggest-recipe handler — EdgeOne Makers Python cloud function.

POST /suggest-recipe
  Body:    { message: string, provider?: "free" | "claude" }
  Returns: { recipes: [{ n, k, meal, min, diet, ing, steps }, ...] }
        or { error: "..." } on failure (missing config, upstream error, etc).

Two providers, chosen per-request by the frontend toggle:
  - "free"   (default) — EdgeOne Makers AI Gateway (OpenAI-compatible), free
             built-in model. Env: AI_GATEWAY_API_KEY, AI_GATEWAY_BASE_URL,
             AI_GATEWAY_MODEL.
  - "claude" — real Anthropic Claude API via the official `anthropic` SDK.
             Env: ANTHROPIC_API_KEY, optional CLAUDE_RECIPE_MODEL (defaults
             to claude-haiku-4-5).
"""

import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler

import requests

# EdgeOne loads each index.py as a top-level module without package context,
# so the parent directory must be on sys.path to import sibling helpers.
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from _logger import create_logger  # noqa: E402

logger = create_logger("suggest-recipe")

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"
DEFAULT_GATEWAY_MODEL = "@makers/deepseek-v4-flash"
DEFAULT_GATEWAY_BASE_URL = "https://ai-gateway.edgeone.link/v1"

RECIPE_SYSTEM_PROMPT = (
    "You are a South Indian home-cooking expert, equally comfortable with "
    "vegetarian and non-vegetarian dishes (chicken, fish, prawns, mutton, eggs). "
    "If the cook mentions meat, fish, prawns, or eggs, or asks for non-vegetarian "
    "food, suggest non-vegetarian South Indian recipes. If they say \"vegetarian\" "
    "or don't mention any meat/fish/egg ingredient, suggest vegetarian ones. "
    "Suggest exactly 2 South Indian recipes that fit what the cook describes."
)

RECIPE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "recipes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "string", "description": "Recipe name"},
                    "k": {"type": "string", "description": "Native name"},
                    "meal": {
                        "type": "string",
                        "enum": ["Tiffin", "Lunch/Dinner", "Snack", "Side", "Sweet"],
                    },
                    "min": {"type": "integer", "description": "Cook time in minutes"},
                    "diet": {"type": "string", "enum": ["veg", "non-veg"]},
                    "ing": {"type": "array", "items": {"type": "string"}},
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["n", "k", "meal", "min", "diet", "ing", "steps"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["recipes"],
    "additionalProperties": False,
}


def _read_body(rfile, headers) -> dict:
    length = int(headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    try:
        return json.loads(rfile.read(length).decode("utf-8")) or {}
    except (ValueError, UnicodeDecodeError):
        return {}


def _suggest_via_gateway(user_message: str) -> list[dict]:
    """Free path — EdgeOne Makers AI Gateway, OpenAI-compatible chat completions."""
    api_key = os.environ.get("AI_GATEWAY_API_KEY", "")
    base_url = os.environ.get("AI_GATEWAY_BASE_URL", DEFAULT_GATEWAY_BASE_URL)
    model = os.environ.get("AI_GATEWAY_MODEL", DEFAULT_GATEWAY_MODEL)

    if not api_key:
        raise RuntimeError(
            "AI_GATEWAY_API_KEY is not configured. Add it in EdgeOne project "
            "settings to use the free model."
        )

    schema_hint = (
        "Respond ONLY with valid JSON (no markdown fences), in this exact shape: "
        '{"recipes": [{"n": "Recipe name", "k": "Native name", '
        '"meal": "Tiffin|Lunch/Dinner|Snack|Side|Sweet", "min": 30, '
        '"diet": "veg|non-veg", "ing": ["item1", "item2"], '
        '"steps": ["step 1", "step 2"]}]}'
    )

    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": f"{RECIPE_SYSTEM_PROMPT} {schema_hint}"},
                {"role": "user", "content": user_message},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(text)
    return parsed.get("recipes", [])


def _suggest_via_claude(user_message: str) -> list[dict]:
    """Paid path — real Claude API via the official Anthropic SDK."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured. Add it in EdgeOne project "
            "settings to use Claude, or switch to the free model."
        )

    model = os.environ.get("CLAUDE_RECIPE_MODEL", DEFAULT_CLAUDE_MODEL)
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=RECIPE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        output_config={"format": {"type": "json_schema", "schema": RECIPE_JSON_SCHEMA}},
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request. Try rephrasing.")

    text = "".join(block.text for block in response.content if block.type == "text")
    parsed = json.loads(text)
    return parsed.get("recipes", [])


class handler(BaseHTTPRequestHandler):
    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=UTF-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        start = time.time()
        body = _read_body(self.rfile, self.headers)
        user_message = str(body.get("message") or "").strip()
        # Provider is a backend-only switch (RECIPE_AI_PROVIDER env var) — not
        # exposed in the frontend. Flip it in EdgeOne project settings to move
        # from the free gateway model to real Claude, no code/UI change needed.
        provider = os.environ.get("RECIPE_AI_PROVIDER", "free").strip().lower()

        if not user_message:
            self._write_json(200, {"error": "Describe what you have first."})
            return

        try:
            if provider == "claude":
                recipes = _suggest_via_claude(user_message)
            else:
                recipes = _suggest_via_gateway(user_message)

            elapsed_ms = int((time.time() - start) * 1000)
            logger.log(f"suggest-recipe: provider={provider} recipes={len(recipes)} in {elapsed_ms}ms")
            self._write_json(200, {"recipes": recipes})

        except Exception as e:
            logger.error(f"suggest-recipe failed: provider={provider} type={type(e).__name__} err={e!r}")
            logger.error(f"traceback:\n{traceback.format_exc()}")
            self._write_json(200, {"error": str(e) or "Couldn't get suggestions — try again."})
