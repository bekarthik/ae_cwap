"""The execution worker as its own process.

`make worker`, and the `worker` service in `docker-compose.yml`. Separate from
`runner.Worker` so that starting one is a module to run rather than a line of
Python to get right in three places.

The log relay matters here specifically. This process emits every run event and
holds no browser connections; the gateway holds the browsers and emits nothing.
Without the relay attached on both sides, the live feed is empty in exactly the
deployment shape the compose file ships.
"""

from __future__ import annotations

import logging

from cwap_common.db import init_db
from cwap_common.diagnostics import configure_logging
from cwap_common.logbus import build_relay, log_bus
from cwap_common.settings import get_settings

from orchestrator.runner import Worker

logger = logging.getLogger("cwap.worker")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s"
    )
    # The cwap namespace gets its own handler (and stops propagating), so its
    # lines appear once whether the host configured the root logger or not.
    configure_logging()
    settings = get_settings()

    init_db()
    log_bus.attach_relay(build_relay())

    logger.info(
        "worker ready — broker=%s database=%s model=%s",
        settings.broker_backend,
        settings.database_url.split("@")[-1],
        settings.llm_provider,
    )
    try:
        Worker().run_forever()
    finally:
        log_bus.attach_relay(None)


if __name__ == "__main__":
    main()
