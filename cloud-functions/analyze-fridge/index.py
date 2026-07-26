"""
Analyze-fridge handler — EdgeOne Makers Python cloud function.

POST /analyze-fridge
  Body:    { image: "<base64>", mimeType: "image/jpeg" }
  Returns: { ingredients: ["tomato", "onion", ...] }
        or { error: "..." } on failure.

Vision (photo -> ingredient list) always goes through real Claude — the free
gateway model is text-only, so there's no free path here. Requires
ANTHROPIC_API_KEY; uses the same CLAUDE_RECIPE_MODEL as suggest-recipe.
"""

import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler

# EdgeOne loads each index.py as a top-level module without package context,
# so the parent directory must be on sys.path to import sibling helpers.
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from _logger import create_logger  # noqa: E402

logger = create_logger("analyze-fridge")

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5"
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_BASE64_CHARS = 7_000_000  # ~5MB decoded, generous margin under Claude's image limit

FRIDGE_PROMPT = (
    "Look at this photo of an open fridge or pantry. List the edible ingredients "
    "and food items you can identify with reasonable confidence — be specific "
    "(e.g. \"tomato\" not \"vegetables\", \"paneer\" or \"eggs\" if visible). "
    "Use South Indian cooking terms where applicable. Only list items you can "
    "actually see; don't guess at what might be inside closed or opaque containers."
)

INGREDIENTS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "ingredients": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ingredients"],
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


def _analyze(image_b64: str, mime_type: str) -> list[str]:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Photo scanning needs Claude's vision — add ANTHROPIC_API_KEY in "
            "EdgeOne project settings to enable it."
        )

    model = os.environ.get("CLAUDE_RECIPE_MODEL", DEFAULT_CLAUDE_MODEL)
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_b64}},
                {"type": "text", "text": FRIDGE_PROMPT},
            ],
        }],
        output_config={"format": {"type": "json_schema", "schema": INGREDIENTS_JSON_SCHEMA}},
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("Claude couldn't process that photo. Try a clearer shot.")

    text = "".join(block.text for block in response.content if block.type == "text")
    parsed = json.loads(text)
    return parsed.get("ingredients", [])


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
        image_b64 = str(body.get("image") or "").strip()
        mime_type = str(body.get("mimeType") or "image/jpeg").strip().lower()

        if not image_b64:
            self._write_json(200, {"error": "No photo received."})
            return
        if mime_type not in ALLOWED_MIME_TYPES:
            self._write_json(200, {"error": f"Unsupported image type: {mime_type}"})
            return
        if len(image_b64) > MAX_BASE64_CHARS:
            self._write_json(200, {"error": "Photo is too large — try a smaller image."})
            return

        try:
            ingredients = _analyze(image_b64, mime_type)
            elapsed_ms = int((time.time() - start) * 1000)
            logger.log(f"analyze-fridge: ingredients={len(ingredients)} in {elapsed_ms}ms")
            self._write_json(200, {"ingredients": ingredients})

        except Exception as e:
            logger.error(f"analyze-fridge failed: type={type(e).__name__} err={e!r}")
            logger.error(f"traceback:\n{traceback.format_exc()}")
            self._write_json(200, {"error": str(e) or "Couldn't read that photo — try again."})
