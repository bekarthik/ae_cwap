"""Real-time Log Streamer (Epic 4).

Every action, external call payload, state change and error becomes a `LogEvent`
that goes to three destinations:

* the `run_logs` table, so the run report is durable and replayable;
* any live WebSocket subscriber in *this* process; and
* a relay, when the deployment runs the worker somewhere else.

The relay is not an optimisation. In `docker-compose` the worker is its own
container, so events are emitted in a process the browser has no connection to —
an in-process fan-out reaches nobody, and "watch it run" silently does nothing in
the topology we actually ship. With Redis already present as the broker, its
pub/sub is the obvious carrier, so a deployment that has one gets a working
stream with no extra moving part.

Sequence numbers are assigned here and are monotonic per run, which is what lets
the UI reconnect mid-run and ask for "everything after seq N" without gaps or
duplicates.

The worker may run on a background thread while subscribers live on the event
loop, so hand-off uses `loop.call_soon_threadsafe`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Protocol

from cwap_contracts.v2 import LogEvent, LogLevel
from sqlalchemy import func

from cwap_common.db import read_only_session, unit_of_work
from cwap_common.idempotency import append_log
from cwap_common.models import RunLog

logger = logging.getLogger("cwap.logbus")

#: Bound on per-run fan-out buffers, so a disconnected browser cannot grow
#: memory without limit.
SUBSCRIBER_BUFFER = 1000

#: The Redis channel every process publishes run events to.
RELAY_CHANNEL = "cwap.run.logs"


class LogRelay(Protocol):
    """Carries log events between processes.

    Narrow on purpose: a deployment that replaces Redis with something else
    implements two methods, and neither the bus nor the worker changes.
    """

    def publish(self, record: LogEvent) -> None: ...

    def listen(self, deliver: Callable[[LogEvent], None]) -> None: ...

    def close(self) -> None: ...


class LogBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequences: dict[str, int] = {}
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
        self._relay: LogRelay | None = None

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

    def transient(
        self,
        run_id: str,
        event: str,
        *,
        node: str | None = None,
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> LogEvent:
        """A live-only event: fanned out and relayed, never written down.

        Token fragments are why this exists. They are worth watching and
        worthless afterwards — the finished text is already on the step — so
        persisting one row per fragment would multiply `run_logs` by a thousand
        to store something the report holds in one piece.

        The sequence number is the current high-water mark rather than a new
        one. A client that reconnects and asks for "everything after N" then
        gets a complete durable history with no gaps where a transient event
        used to be, because a transient event never consumed a number.

        That reused number is exactly why `transient: true` is stamped into the
        data: a subscriber discards anything at or below the sequence it has
        already replayed, which would otherwise discard every one of these.
        """
        record = LogEvent(
            run_id=run_id,
            seq=self._current_seq(run_id),
            node=node,
            level=LogLevel.INFO,
            event=event,
            message=message[:2000],
            data={**_scrub(data or {}), "transient": True},
        )
        self._fan_out(record)
        if self._relay is not None:
            with suppress(Exception):  # a live tail must never fail a run
                self._relay.publish(record)
        return record

    def _current_seq(self, run_id: str) -> int:
        with self._lock:
            if run_id not in self._sequences:
                self._sequences[run_id] = self._highest_persisted(run_id)
            return self._sequences[run_id]

    def publish(self, record: LogEvent) -> None:
        """Persist, fan out locally, and relay to other processes."""
        with unit_of_work() as session:
            append_log(session, record)
        self._fan_out(record)

        if self._relay is not None:
            try:
                self._relay.publish(record)
            except Exception:  # noqa: BLE001 - a live tail must not fail a run
                logger.warning("could not relay a log event", exc_info=True)

    def deliver(self, record: LogEvent) -> None:
        """Hand a record from *another* process to this one's subscribers.

        Deliberately does not persist: the process that emitted it already did,
        and writing again would double every line in the run report.
        """
        self._fan_out(record)

    def attach_relay(self, relay: LogRelay | None) -> None:
        """Wire in a cross-process carrier, and start listening on it."""
        if self._relay is not None:
            self._relay.close()
        self._relay = relay
        if relay is not None:
            relay.listen(self.deliver)

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


class RedisLogRelay:
    """Cross-process fan-out over Redis pub/sub.

    One channel for every run rather than one per run: a gateway serves many
    browsers watching different runs, and re-subscribing per run would mean
    connection churn on every page load for no benefit at this volume. Each
    process filters to the runs it actually has subscribers for.
    """

    def __init__(self, url: str) -> None:
        import redis  # noqa: PLC0415 - optional dependency

        self._client = redis.Redis.from_url(url)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def publish(self, record: LogEvent) -> None:
        self._client.publish(RELAY_CHANNEL, record.model_dump_json())

    def listen(self, deliver: Callable[[LogEvent], None]) -> None:
        if self._thread is not None:
            return

        pubsub = self._client.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(RELAY_CHANNEL)

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    message = pubsub.get_message(timeout=1.0)
                except Exception:  # noqa: BLE001 - a dropped tail must not crash a process
                    logger.warning("log relay read failed; retrying", exc_info=True)
                    self._stop.wait(1.0)
                    continue
                if not message:
                    continue
                try:
                    deliver(LogEvent.model_validate(json.loads(message["data"])))
                except Exception:  # noqa: BLE001 - one bad frame is not fatal
                    logger.warning("could not decode a relayed log event", exc_info=True)

        self._thread = threading.Thread(target=loop, name="cwap-log-relay", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with suppress(Exception):
            self._client.close()


def build_relay() -> LogRelay | None:
    """A relay when the deployment needs one, None when it does not.

    "Needs one" means the worker is a separate process, which is exactly what
    choosing the Redis broker says. A single-process deployment already shares
    the bus, so a relay would be pure overhead and a second failure mode.
    """
    from cwap_common.settings import get_settings  # noqa: PLC0415 - read at call time

    settings = get_settings()
    if settings.broker_backend.strip().lower() != "redis":
        return None

    try:
        return RedisLogRelay(settings.redis_url)
    except Exception:  # noqa: BLE001 - a missing tail must not stop the platform
        logger.warning(
            "could not start the cross-process log relay; the run report will "
            "still be complete, but the live feed will not update",
            exc_info=True,
        )
        return None


#: Process-wide bus. Services import this rather than constructing their own,
#: so the worker thread and the WebSocket handler share subscriber state.
log_bus = LogBus()
