"""Passing an image to a model that can actually look at one.

`supports_vision` was detected, stored, reported through the API and shown on
the model picker, and no request ever carried an image. A step whose input was
`https://example.com/chart.png` sent the model that sentence — twenty-eight
characters describing a picture the model was perfectly capable of reading.

The fix is small and the two constraints on it are what make it worth its own
module:

* **Only when the model can see.** Sending image blocks to a text-only model is
  a 400 on most servers and silently dropped content on the rest, so this is
  gated on the capability the catalogue already tracks.
* **Only what looks like an image.** A URL is not an image because it is a URL.
  Guessing wrong turns a link the user wanted quoted into a fetch the model
  cannot complete, so the match is on an image extension, and the URL stays in
  the text as well — the model sees the picture *and* knows where it came from.

Nothing here downloads anything. The URL is handed to the provider, which hands
it to the model's own fetcher; a platform that pulled the bytes itself would be
making an outbound request on a user-supplied address, which is the SSRF surface
the egress allow-list exists to close.
"""

from __future__ import annotations

import re
from typing import Any

#: What counts as an image. Extension-based on purpose: a URL is not an image
#: because it is a URL, and a wrong guess sends a model somewhere it cannot go.
_IMAGE_URL = re.compile(
    r"https?://[^\s<>\"')]+\.(?:png|jpe?g|gif|webp|bmp)(?:\?[^\s<>\"')]*)?",
    re.IGNORECASE,
)

#: More than this in one message and the request is mostly pictures; several
#: megabytes of images per step is a cost decision, not a formatting one.
MAX_IMAGES = 6


def find(text: str) -> list[str]:
    """Image URLs in a piece of text, in order, without duplicates."""
    seen: list[str] = []
    for match in _IMAGE_URL.findall(text or ""):
        if match not in seen:
            seen.append(match)
    return seen[:MAX_IMAGES]


def openai_content(text: str) -> str | list[dict[str, Any]]:
    """One user message for an OpenAI-compatible server.

    Returns the plain string when there is nothing to look at, so a request
    against a text model is byte-for-byte what it always was.
    """
    images = find(text)
    if not images:
        return text

    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    blocks.extend({"type": "image_url", "image_url": {"url": url}} for url in images)
    return blocks


def anthropic_content(text: str) -> str | list[dict[str, Any]]:
    """The same message for Claude, which names the block type differently."""
    images = find(text)
    if not images:
        return text

    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    blocks.extend(
        {"type": "image", "source": {"type": "url", "url": url}} for url in images
    )
    return blocks
