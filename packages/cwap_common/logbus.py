"""Real-time Log Streamer (Epic 4).

Every action, external call payload, state change and error becomes a `LogEvent`
that goes to two destinations at once:

* the `run_logs` table, so the run report is durable and replayable; and
* any live WebSocket subscriber, so the user watches it happen.

Sequence numbers are assigned here and are monotonic per run, which is what lets
the UI reconnect mid-run and ask for "everything after seq N" without gaps or
duplicates.

The worker may run on a background thread while subscribers live on the event
loop, so hand-off uses `loop.call_soon_threadsafe`.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import suppress
from typing import Any

from cwap_contracts import LogEvent, LogLevel
from sqlalchemy import func

from cwap_common.db import read_only_session, unit_of_work
from cwap_common.idempotency import append_log
from cwap_common.models import RunLog

#: Bound on per-run fan-out buffers, so a disconnected browser cannot grow
#: memory without limit.
SUBSCRIBER_BUFFER = 1000


class LogBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequences: dict[str, int] = {}
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}

    # ---- sequencing ----------------------------------------------------

    def _next_seq(self, run_id: str) -> int:
        with self._lock:
            if run_id not in self._sequences:
                self._sequences[run_id] = self._highest_persisted(run_id)
            self._sequences[run_id] += 1
            return self._sequences[run_id]

    @staticmethod
    def _highest_persisted(run_id: str) -> int:
        with read_only_session() as session:
            highest = (
                session.query(func.max(RunLog.seq)).filter(RunLog.run_id == run_id).scalar()
            )
        return int(highest or 0)

    # ---- publishing ----------------------------------------------------

    def emit(
        self,
        run_id: str,
        event: str,
        *,
        node: str | None = None,
        level: LogLevel = LogLevel.INFO,
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> LogEvent:
        record = LogEvent(
            run_id=run_id,
            seq=self._next_seq(run_id),
            node=node,
            level=level,
            event=event,
            message=message[:2000],
            data=_scrub(data or {}),
        )
        self.publish(record)
        return record

    def publish(self, record: LogEvent) -> None:
        with unit_of_work() as session:
            append_log(session, record)
        self._fan_out(record)

    def _fan_out(self, record: LogEvent) -> None:
        with self._lock:
            targets = list(self._subscribers.get(record.run_id, ()))

        for loop, queue in targets:
            try:
                if loop.is_closed():
                    continue
                loop.call_soon_threadsafe(_offer, queue, record)
            except RuntimeError:  # pragma: no cover - loop shut down mid-publish
                continue

    # ---- subscription --------------------------------------------------

    def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=SUBSCRIBER_BUFFER)
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subscribers.setdefault(run_id, []).append((loop, queue))
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            remaining = [entry for entry in self._subscribers.get(run_id, []) if entry[1] is not queue]
            if remaining:
                self._subscribers[run_id] = remaining
            else:
                self._subscribers.pop(run_id, None)

    def subscriber_count(self, run_id: str) -> int:
        with self._lock:
            return len(self._subscribers.get(run_id, ()))

    # ---- replay --------------------------------------------------------

    @staticmethod
    def history(run_id: str, after_seq: int = 0, limit: int = 2000) -> list[LogEvent]:
        with read_only_session() as session:
            rows = (
                session.query(RunLog)
                .filter(RunLog.run_id == run_id, RunLog.seq > after_seq)
                .order_by(RunLog.seq)
                .limit(limit)
                .all()
            )
        return [
            LogEvent(
                run_id=row.run_id,
                seq=row.seq,
                timestamp=row.timestamp,
                node=row.node,
                level=LogLevel(row.level),
                event=row.event,
                message=row.message,
                data=row.data or {},
            )
            for row in rows
        ]

    def forget(self, run_id: str) -> None:
        """Drop in-memory sequence state for a finished run."""
        with self._lock:
            self._sequences.pop(run_id, None)


def _offer(queue: asyncio.Queue, record: LogEvent) -> None:
    """Never block the producer. A subscriber that cannot keep up drops its
    oldest event and carries on — the durable copy is already in Postgres."""
    if queue.full():
        # Race with the consumer: it drained the queue between the check and
        # here. Nothing to drop, so carry on.
        with suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    # Race with another producer that refilled it. The durable copy is already
    # in the database, so dropping the live event is acceptable.
    with suppress(asyncio.QueueFull):
        queue.put_nowait(record)


#: Header/param names whose values must never reach a log line or the browser.
SENSITIVE_KEYS = frozenset(
    {"authorization", "api_key", "apikey", "password", "secret", "token", "x-api-key", "cookie"}
)


def _scrub(data: dict[str, Any]) -> dict[str, Any]:
    """Redact credentials before they are persisted or streamed.

    The log feed is deliberately verbose — it records external call payloads —
    so redaction has to happen at the one place every event passes through.
    """
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_KEYS:
            cleaned[key] = "***redacted***"
        elif isinstance(value, dict):
            cleaned[key] = _scrub(value)
        elif isinstance(value, list):
            cleaned[key] = [_scrub(v) if isinstance(v, dict) else v for v in value]
        else:
            cleaned[key] = value
    return cleaned


#: Process-wide bus. Services import this rather than constructing their own,
#: so the worker thread and the WebSocket handler share subscriber state.
log_bus = LogBus()
