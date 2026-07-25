"""The Schema Registry.

Mandate §1: "Any change to a service output requires an approved version bump in
this centralized contract."

We make that mechanical rather than cultural. Every published contract is
registered here under `Name@version`, and its JSON Schema is fingerprinted into
`contracts.lock.json`. `tests/test_contract_registry.py` fails the build if a
live fingerprint drifts from the lock — so the only way to change a contract's
shape is to bump its version and re-approve the lock in review.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cwap_contracts.errors import SchemaNotRegistered

LOCK_PATH = Path(__file__).with_name("contracts.lock.json")

#: "ContractName@version" -> model class
CONTRACT_REGISTRY: dict[str, type[BaseModel]] = {}


def register(model: type[BaseModel], *, version: str) -> type[BaseModel]:
    """Publish a model as a boundary-crossing contract."""
    key = f"{model.__name__}@{version}"
    existing = CONTRACT_REGISTRY.get(key)
    if existing is not None and existing is not model:
        raise ValueError(f"contract '{key}' is already registered to {existing!r}")
    CONTRACT_REGISTRY[key] = model
    return model


def resolve(name: str, version: str) -> type[BaseModel]:
    """Look up the canonical model for a contract name + version."""
    key = f"{name}@{version}"
    try:
        return CONTRACT_REGISTRY[key]
    except KeyError as exc:
        raise SchemaNotRegistered(
            f"no contract registered as '{key}'",
            detail={"known": sorted(CONTRACT_REGISTRY)},
        ) from exc


def latest_version(name: str) -> str:
    """Highest registered version for a contract name, e.g. 'v1'."""
    versions = [
        key.split("@", 1)[1] for key in CONTRACT_REGISTRY if key.split("@", 1)[0] == name
    ]
    if not versions:
        raise SchemaNotRegistered(f"no contract registered under the name '{name}'")
    return max(versions, key=lambda v: int(v.lstrip("v")))


def fingerprint(model: type[BaseModel]) -> str:
    """Stable sha256 over the model's JSON Schema.

    Key order is normalised so that a pure reordering of field declarations does
    not look like a breaking change, while any added/removed/retyped field does.
    """
    schema: dict[str, Any] = model.model_json_schema(mode="serialization")
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def lock_snapshot() -> dict[str, str]:
    """Current fingerprints for every registered contract."""
    return {key: fingerprint(model) for key, model in sorted(CONTRACT_REGISTRY.items())}


def read_lock() -> dict[str, str]:
    if not LOCK_PATH.exists():
        return {}
    return json.loads(LOCK_PATH.read_text())


def write_lock() -> dict[str, str]:
    """Re-approve the lock. Run deliberately, as part of a version bump."""
    snapshot = lock_snapshot()
    LOCK_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    return snapshot


def _register_v1() -> None:
    from cwap_contracts import v1

    for model in (
        v1.DiagnosisResult,
        v1.GoalIntakeRequest,
        v1.IngestRequest,
        v1.JobContext,
        v1.KnowledgeHandle,
        v1.LogEvent,
        v1.NextStepDefinition,
        v1.PermissionRequirement,
        v1.RetrievalRequest,
        v1.RetrievalResult,
        v1.ScaffoldResponse,
        v1.StepOutputContext,
        v1.WorkflowGraph,
        v1.WorkflowJobPayload,
    ):
        register(model, version="v1")


_register_v1()


if __name__ == "__main__":  # pragma: no cover - maintenance entry point
    approved = write_lock()
    print(f"wrote {len(approved)} contract fingerprints to {LOCK_PATH}")
    for key, digest in approved.items():
        print(f"  {key:<40} {digest[:16]}")
