"""Shared runtime foundation used by every service.

Deliberately thin and dependency-light: persistence, transactions, idempotency,
transport, the dual gateway, and the log bus. Business logic lives in the
services; this package only provides the guarantees they are all required to
uphold.
"""

from cwap_common.authz import (
    AuthorizationClient,
    AuthorizationDecision,
    LocalAuthorizationService,
)
from cwap_common.broker import Broker, InMemoryBroker, get_broker, set_broker
from cwap_common.contract_gateway import (
    ConsumerWrapper,
    DeadLetterSink,
    ProducerWrapper,
    validate_payload,
)
from cwap_common.db import (
    configure,
    dispose,
    get_engine,
    get_session_factory,
    init_db,
    read_only_session,
    unit_of_work,
)
from cwap_common.idempotency import (
    ClaimResult,
    IdempotencyGate,
    append_log,
    commit_step_output,
)
from cwap_common.logbus import LogBus, log_bus
from cwap_common.settings import Settings, get_settings, reset_settings_cache

__all__ = [
    "AuthorizationClient",
    "AuthorizationDecision",
    "Broker",
    "ClaimResult",
    "ConsumerWrapper",
    "DeadLetterSink",
    "IdempotencyGate",
    "InMemoryBroker",
    "LocalAuthorizationService",
    "LogBus",
    "ProducerWrapper",
    "Settings",
    "append_log",
    "commit_step_output",
    "configure",
    "dispose",
    "get_broker",
    "get_engine",
    "get_session_factory",
    "get_settings",
    "init_db",
    "log_bus",
    "read_only_session",
    "reset_settings_cache",
    "set_broker",
    "unit_of_work",
    "validate_payload",
]
