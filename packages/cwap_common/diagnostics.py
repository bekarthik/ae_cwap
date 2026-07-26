"""Making the platform's own log lines actually come out of a container.

Uvicorn configures its own loggers and nothing else. The `cwap.*` loggers had
no handler, so everything below WARNING was silently dropped in exactly the
deployment shape the compose file ships — and the WARNINGs that did escape came
through Python's last-resort handler, bare, with no timestamp.

The cost of that was not hypothetical. Three rounds of debugging a failing MCP
connection were carried out against a container whose only visible output was
uvicorn's access log, while the connector was writing down precisely what it
tried, what answered, and how long each step took — to a logger nobody had
wired to anything.
"""

from __future__ import annotations

import logging
import os
import sys

#: Everything the platform logs lives under this namespace.
NAMESPACE = "cwap"


def configure_logging() -> None:
    """Give the platform's loggers a handler, once, without touching anyone else's.

    Scoped to the `cwap` namespace rather than the root logger on purpose: the
    root belongs to whoever is hosting us (uvicorn, Celery, a test runner), and
    configuring it from library-ish code is how log lines end up duplicated or
    reformatted out from under the host.

    Idempotent, so the gateway lifespan and the worker entry can both call it
    and a process that is somehow both only gets one handler.
    """
    logger = logging.getLogger(NAMESPACE)
    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("CWAP_LOG_LEVEL", "INFO").upper())
    # Our handler is the whole story for this namespace. Propagating too would
    # print every line twice in any process that also configures the root.
    logger.propagate = False
