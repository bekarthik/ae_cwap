"""Turning a stream of fragments into something a browser can keep up with.

A model emits tokens faster than any UI needs them, and every fragment that
reaches the log bus costs a fan-out and a websocket frame. Forwarding each one
would spend most of a run's event budget on the word "the".

So fragments are coalesced: pushed when enough characters have accumulated or
enough time has passed, whichever comes first. The interval is what makes it
feel live — text arriving two or three times a second reads as typing — and the
character bound is what keeps a fast hosted model from pushing hundreds of times
a second between ticks.

Content and reasoning are buffered separately. They are two different things to
look at, and interleaving them into one string would produce a paragraph that is
half answer and half thinking with no way to tell which is which.
"""

from __future__ import annotations

import time
from collections.abc import Callable

#: Roughly three pushes a second. Fast enough to read as typing, slow enough
#: that a long answer is tens of events rather than thousands.
INTERVAL_SECONDS = 0.35

#: Push early once this much text is waiting, so a burst is not held back by
#: the clock.
MAX_PENDING_CHARS = 160


class LiveText:
    """A delta sink that forwards in readable pieces instead of tokens.

    Call it like any sink; call `flush()` when the answer is finished so the
    tail is not left sitting in the buffer, which would show a preview that
    stops a few words short of what the model actually said.
    """

    def __init__(
        self,
        publish: Callable[[str, str], None],
        *,
        interval: float = INTERVAL_SECONDS,
        max_chars: int = MAX_PENDING_CHARS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._publish = publish
        self._interval = interval
        self._max_chars = max_chars
        self._clock = clock
        self._pending: dict[str, list[str]] = {}
        self._last: dict[str, float] = {}

    def __call__(self, kind: str, text: str) -> None:
        if not text:
            return
        buffer = self._pending.setdefault(kind, [])
        buffer.append(text)

        now = self._clock()
        due = now - self._last.get(kind, 0.0) >= self._interval
        full = sum(len(part) for part in buffer) >= self._max_chars
        if due or full:
            self._push(kind, now)

    def flush(self) -> None:
        """Send whatever is still buffered, for every kind."""
        now = self._clock()
        for kind in list(self._pending):
            self._push(kind, now)

    def _push(self, kind: str, now: float) -> None:
        text = "".join(self._pending.pop(kind, ()))
        if not text:
            return
        self._last[kind] = now
        self._publish(kind, text)
