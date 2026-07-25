"""Encryption for credentials the platform has to keep.

Switching model provider from the browser means the platform stores an API key,
which is a meaningful change in what a database dump is worth. So keys are
encrypted at rest with a key derived from `CWAP_SECRET_KEY` (falling back to the
JWT secret, which any real deployment already has to set), and are never
returned by any API — a caller can learn *that* a key is configured, never what
it is.

Three things this deliberately is not:

* **Not a secrets manager.** An operator who wants Vault or AWS Secrets Manager
  should set the key through the environment and leave the per-tenant field
  empty; env configuration still wins where a tenant has not overridden it.
* **Not protection from a compromised application server.** The process can
  decrypt by definition. It protects a leaked backup, a stolen replica, or a
  careless `SELECT *` — which is most of what actually happens.
* **Not a reason to log the plaintext.** `logbus._scrub` already redacts
  credential-shaped keys; this module never logs at all.
"""

from __future__ import annotations

import base64
import hashlib

from cwap_common.settings import get_settings

#: Marks ciphertext produced here, so a plaintext value written before this
#: existed (or by hand, in a migration) is recognisable rather than being fed to
#: the decrypter and failing.
PREFIX = "enc:v1:"


class SecretError(RuntimeError):
    """A stored secret could not be decrypted."""


def _fernet():
    from cryptography.fernet import Fernet  # noqa: PLC0415 - optional at import time

    settings = get_settings()
    material = (settings.secret_key or settings.jwt_secret).encode()
    # Fernet wants exactly 32 url-safe base64 bytes; SHA-256 of the configured
    # secret gives that deterministically, so the same deployment secret always
    # yields the same key and no separate key file has to be managed.
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(material).digest()))


def encrypt(value: str) -> str:
    """Encrypt a credential for storage. An empty value stays empty."""
    if not value:
        return ""
    return PREFIX + _fernet().encrypt(value.encode()).decode()


def decrypt(stored: str) -> str:
    """Recover a credential. Values without the marker are returned as-is.

    That tolerance is deliberate: it lets an operator seed a row by hand and
    lets an older plaintext row keep working, instead of a deployment failing
    with an opaque error at the first model call.
    """
    if not stored:
        return ""
    if not stored.startswith(PREFIX):
        return stored

    from cryptography.fernet import InvalidToken  # noqa: PLC0415

    try:
        return _fernet().decrypt(stored[len(PREFIX) :].encode()).decode()
    except InvalidToken as exc:
        raise SecretError(
            "a stored credential could not be decrypted — CWAP_SECRET_KEY has "
            "changed since it was saved. Re-enter the credential, or restore the "
            "previous secret."
        ) from exc


def is_encrypted(stored: str) -> bool:
    return stored.startswith(PREFIX)
