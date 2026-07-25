"""Message transport.

Two interchangeable backends behind one interface:

* `InMemoryBroker` — the default. Lets the whole platform (API + worker + canvas)
  run with no infrastructure, and makes the execution tests deterministic.
* `RedisBroker` — the production path, list-based (`LPUSH`/`BRPOP`), which is the
  same primitive Celery's Redis transport uses.

Neither backend knows anything about contracts. Validation lives one layer up in
`contract_gateway`, so it is impossible to publish a payload that skipped it.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Protocol

from cwap_common.settings import get_settings


class Broker(Protocol):
    def publish(self, queue: str, body: str) -> None: ...

    def consume(self, queue: str, timeout: float = 0.0) -> str | None: ...

    def depth(self, queue: str) -> int: ...

    def drain(self, queue: str) -> list[str]: ...


class InMemoryBroker:
    """Thread-safe FIFO queues held in process memory."""

    def __init__(self) -> None:
        self._queues: dict[str, deque[str]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def publish(self, queue: str, body: str) -> None:
        with self._not_empty:
            self._queues[queue].append(body)
            self._not_empty.notify()

    def consume(self, queue: str, timeout: float = 0.0) -> str | None:
        with self._not_empty:
            if not self._queues[queue] and timeout > 0:
                self._not_empty.wait(timeout)
            if not self._queues[queue]:
                return None
            return self._queues[queue].popleft()

    def depth(self, queue: str) -> int:
        with self._lock:
            return len(self._queues[queue])

    def drain(self, queue: str) -> list[str]:
        with self._lock:
            items = list(self._queues[queue])
            self._queues[queue].clear()
            return items

    def reset(self) -> None:
        with self._lock:
            self._queues.clear()


class RedisBroker:
    """Redis list transport. Imported lazily so `redis` stays an extra."""

    def __init__(self, url: str) -> None:
        import redis  # noqa: PLC0415 - optional dependency

        self._client = redis.Redis.from_url(url, decode_responses=True)

    def publish(self, queue: str, body: str) -> None:
        self._client.lpush(queue, body)

    def consume(self, queue: str, timeout: float = 0.0) -> str | None:
        if timeout > 0:
            item = self._client.brpop(queue, timeout=int(timeout) or 1)
            return item[1] if item else None
        return self._client.rpop(queue)

    def depth(self, queue: str) -> int:
        return int(self._client.llen(queue))

    def drain(self, queue: str) -> list[str]:
        items = self._client.lrange(queue, 0, -1) or []
        self._client.delete(queue)
        return list(reversed(items))


_broker: Broker | None = None
_broker_lock = threading.Lock()


def get_broker() -> Broker:
    global _broker
    if _broker is None:
        with _broker_lock:
            if _broker is None:
                settings = get_settings()
                if settings.broker_backend == "redis":
                    _broker = RedisBroker(settings.redis_url)
                else:
                    _broker = InMemoryBroker()
    return _broker


def set_broker(broker: Broker | None) -> None:
    """Swap the process broker. Used by test fixtures and by the dev runner."""
    global _broker
    with _broker_lock:
        _broker = broker
