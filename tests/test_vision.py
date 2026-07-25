"""A model that can look at a picture should be given the picture.

`supports_vision` was detected, stored, returned by the API and shown on the
model picker, and no request ever carried an image: a step whose input was a URL
ending in `.png` sent the model that sentence. This is the same defect the
thinking work found one layer over — a capability the platform advertised and
never used.

Two constraints do the real work here. Sending image blocks to a text model is a
400 on most servers, so it is gated on the capability; and a URL is not an image
because it is a URL, so the match is on an extension and the URL stays in the
text as well.
"""

from __future__ import annotations

import json

import httpx
import pytest
from llm_proxy import vision
from llm_proxy.client import ChatMessage, OpenAICompatibleProvider

CHART = "https://example.com/quarterly-chart.png"


@pytest.fixture
def patched_httpx(monkeypatch):
    def install(handler):
        transport = httpx.MockTransport(handler)
        real_client = httpx.Client

        def factory(**kwargs):
            kwargs.pop("transport", None)
            return real_client(transport=transport, **kwargs)

        monkeypatch.setattr(httpx, "Client", factory)

    return install


def answer(text: str = "a bar chart"):
    return httpx.Response(
        200,
        json={
            "model": "m",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        },
    )


def seeing(model: str = "gemma3") -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(
        provider="ollama", base_url="http://localhost:11434/v1", model=model
    )
    assert provider.capabilities.supports_vision, "fixture model must be a vision model"
    return provider


def blind(model: str = "llama3.1") -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(
        provider="ollama", base_url="http://localhost:11434/v1", model=model
    )
    assert not provider.capabilities.supports_vision
    return provider


class TestAnImageBecomesAnImage:
    def test_a_url_in_the_prompt_is_sent_as_a_picture(self, patched_httpx):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return answer()

        patched_httpx(handler)
        seeing().complete(f"What does this show? {CHART}")

        blocks = seen["messages"][-1]["content"]
        assert {block["type"] for block in blocks} == {"text", "image_url"}
        assert blocks[1]["image_url"]["url"] == CHART

    def test_the_url_stays_in_the_text_too(self, patched_httpx):
        """The model sees the picture *and* knows where it came from."""
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return answer()

        patched_httpx(handler)
        seeing().complete(f"Compare {CHART} with last year")

        text = seen["messages"][-1]["content"][0]["text"]
        assert CHART in text

    def test_a_conversation_turn_carries_images_as_well(self, patched_httpx):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return answer()

        patched_httpx(handler)
        seeing().converse([ChatMessage(role="user", content=f"read {CHART}")])

        assert seen["messages"][-1]["content"][1]["type"] == "image_url"

    def test_several_images_all_arrive(self):
        second = "https://example.com/b.jpg"
        blocks = vision.openai_content(f"{CHART} and {second}")

        assert [block.get("image_url", {}).get("url") for block in blocks[1:]] == [
            CHART,
            second,
        ]

    def test_claude_gets_its_own_block_shape(self):
        blocks = vision.anthropic_content(f"look at {CHART}")

        assert blocks[1] == {"type": "image", "source": {"type": "url", "url": CHART}}


class TestWhatIsDeliberatelyNotSent:
    def test_a_text_model_gets_a_plain_string(self, patched_httpx):
        """Image blocks to a text-only model are a 400 on most servers."""
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return answer()

        patched_httpx(handler)
        blind().complete(f"What does this show? {CHART}")

        assert seen["messages"][-1]["content"] == f"What does this show? {CHART}"

    def test_a_url_that_is_not_an_image_is_left_alone(self):
        """Guessing wrong turns a link the user wanted quoted into a fetch."""
        assert vision.find("see https://example.com/report") == []
        assert vision.find("https://example.com/page.html") == []

    def test_a_query_string_does_not_hide_the_extension(self):
        assert vision.find("https://cdn.example.com/a.png?width=200") == [
            "https://cdn.example.com/a.png?width=200"
        ]

    def test_the_same_image_twice_is_sent_once(self):
        assert vision.find(f"{CHART} and again {CHART}") == [CHART]

    def test_a_wall_of_images_is_capped(self):
        text = " ".join(f"https://example.com/{index}.png" for index in range(20))

        assert len(vision.find(text)) == vision.MAX_IMAGES

    def test_an_assistant_turn_is_never_rewritten(self):
        """Only what the user supplied is scanned; the model's own words are
        passed back exactly as it said them."""
        from llm_proxy.client import _to_openai_messages

        out = _to_openai_messages(
            [ChatMessage(role="assistant", content=f"I made {CHART}")], None, sees=True
        )

        assert out[0]["content"] == f"I made {CHART}"
