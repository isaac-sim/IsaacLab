# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Append-only baseline store, addressed by URI.

TODO: decide backend and layout to finalize implementation.

The store is the only module that performs I/O or touches credentials.

Reads and writes share one code path. Credentials are never passed as arguments:
each backend resolves the ambient identity (an OIDC-assumed role on AWS, a managed
identity on Azure), so read and write differ **only** in which permission that
identity holds. Isolation is therefore enforced by the cloud trust policy scoped
to ``refs/heads/develop`` for writes.

Supported URIs:
    ``file:///path/to/baselines``   local directory (development, tests, CI dry runs)
    ``s3://bucket/prefix``          AWS S3 (requires ``boto3``)
    ``az://container/prefix``       Azure Blob Storage (requires ``azure-storage-blob``)

Layout: one newline-delimited JSON file per contract hash, appended to. Rows are
never mutated or deleted, so a bad row is recoverable and history stays auditable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .metrics import METRICS, PerfSmokeError, mapping, number


@dataclass(frozen=True)
class BaselineRow:
    """One recorded develop measurement for a single contract.

    Attributes:
        contract: Serialised :class:`~tools.perf_smoke.contract.Contract`.
        contract_hash: Digest of ``contract``, used as the storage key.
        metrics: Measured value per metric name.
        commit: Develop commit the measurement was taken on.
        timestamp: ISO-8601 UTC timestamp of the run.
        run_id: CI run identifier, for tracing a row back to its logs.
    """

    contract: dict[str, Any]
    contract_hash: str
    metrics: dict[str, float]
    commit: str
    timestamp: str
    run_id: str

    def as_dict(self) -> dict[str, Any]:
        """Return the row as a plain, JSON-serialisable dict."""
        return {
            "contract": self.contract,
            "contract_hash": self.contract_hash,
            "metrics": dict(self.metrics),
            "commit": self.commit,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
        }


def parse_row(data: Any, source: str) -> BaselineRow:
    """Parse and validate one stored row.

    Args:
        data: Decoded JSON object.
        source: Human-readable origin, used in error messages.

    Returns:
        The validated row.

    Raises:
        PerfSmokeError: If required fields are missing or malformed. A stored row
            that cannot be parsed is corruption, not absence, and must fail loudly.
    """
    payload = mapping(data, source)
    stored_metrics = mapping(payload.get("metrics"), f"{source}.metrics")
    values: dict[str, float] = {}
    for metric in METRICS:
        # Rows written before a metric was added simply lack it.
        if metric.name not in stored_metrics:
            continue
        values[metric.name] = number(stored_metrics[metric.name], f"{source}.metrics.{metric.name}")
    if not values:
        raise PerfSmokeError(f"{source}.metrics contains no recognised metric")

    contract_hash = payload.get("contract_hash")
    if not isinstance(contract_hash, str) or not contract_hash.strip():
        raise PerfSmokeError(f"{source}.contract_hash must be a non-empty string")

    return BaselineRow(
        contract=mapping(payload.get("contract"), f"{source}.contract"),
        contract_hash=contract_hash,
        metrics=values,
        commit=str(payload.get("commit", "")),
        timestamp=str(payload.get("timestamp", "")),
        run_id=str(payload.get("run_id", "")),
    )


class _Backend:
    """Minimal object-store interface for any backend"""

    def read_text(self, key: str) -> str | None:
        """Return the object's contents, or ``None`` when it does not exist."""
        raise NotImplementedError

    def append_text(self, key: str, text: str) -> None:
        """Append ``text`` to the object, creating it when absent."""
        raise NotImplementedError


class _FileBackend(_Backend):
    """Local-filesystem backend for tests and dry runs."""

    def __init__(self, root: str) -> None:
        from pathlib import Path

        self._root = Path(root)

    def read_text(self, key: str) -> str | None:
        path = self._root / key
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def append_text(self, key: str, text: str) -> None:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)


# Sample backends
# TODO: decide on backend storage solution.
class _S3Backend(_Backend):
    """AWS S3 backend. Credentials come from the ambient (OIDC-assumed) identity."""

    def __init__(self, bucket: str, prefix: str) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on runner image
            raise PerfSmokeError("s3:// baselines require boto3 to be installed") from exc
        self._client = boto3.client("s3")
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def read_text(self, key: str) -> str | None:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=self._key(key))
        except self._client.exceptions.NoSuchKey:
            return None
        return response["Body"].read().decode("utf-8")

    def append_text(self, key: str, text: str) -> None:
        # S3 objects are immutable, so append is read-modify-write. Safe because each
        # matrix combination owns a distinct contract hash and therefore a distinct
        # key: the concurrent develop jobs never touch the same object.
        existing = self.read_text(key) or ""
        self._client.put_object(Bucket=self._bucket, Key=self._key(key), Body=(existing + text).encode("utf-8"))


class _AzureBackend(_Backend):
    """Azure Blob Storage backend. Credentials come from the ambient managed identity."""

    def __init__(self, container: str, prefix: str) -> None:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:  # pragma: no cover - depends on runner image
            raise PerfSmokeError("az:// baselines require azure-storage-blob and azure-identity") from exc
        import os

        account_url = os.environ.get("PERF_BASELINE_AZURE_ACCOUNT_URL")
        if not account_url:
            raise PerfSmokeError("az:// baselines require PERF_BASELINE_AZURE_ACCOUNT_URL")
        service = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
        self._container = service.get_container_client(container)
        self._prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def read_text(self, key: str) -> str | None:
        blob = self._container.get_blob_client(self._key(key))
        if not blob.exists():
            return None
        return blob.download_blob().readall().decode("utf-8")

    def append_text(self, key: str, text: str) -> None:
        existing = self.read_text(key) or ""
        self._container.get_blob_client(self._key(key)).upload_blob((existing + text).encode("utf-8"), overwrite=True)


def _backend(uri: str) -> _Backend:
    """Resolve a store URI to its backend."""
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        return _FileBackend(parsed.path or uri)
    if parsed.scheme == "s3":
        return _S3Backend(parsed.netloc, parsed.path)
    if parsed.scheme == "az":
        return _AzureBackend(parsed.netloc, parsed.path)
    raise PerfSmokeError(f"Unsupported baseline URI scheme {parsed.scheme!r}; expected file://, s3:// or az://")


def _key_for(contract_hash: str) -> str:
    return f"{contract_hash}.ndjson"


def read(uri: str, contract_hash: str, limit: int) -> list[BaselineRow]:
    """Return the most recent rows recorded for ``contract_hash``.

    Args:
        uri: Baseline store URI.
        contract_hash: Digest identifying poolable runs.
        limit: Maximum number of rows to return, newest last.

    Returns:
        Up to ``limit`` rows in stored (chronological) order. Empty when the
        contract has never been recorded.

    Raises:
        PerfSmokeError: If a stored row exists but cannot be parsed.
    """
    if limit <= 0:
        raise PerfSmokeError("limit must be a positive integer")
    text = _backend(uri).read_text(_key_for(contract_hash))
    if text is None:
        return []
    rows: list[BaselineRow] = []
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PerfSmokeError(f"{contract_hash}.ndjson line {index + 1} is not valid JSON: {exc}") from exc
        rows.append(parse_row(decoded, f"{contract_hash}.ndjson line {index + 1}"))
    return rows[-limit:]


def write(uri: str, row: BaselineRow) -> None:
    """Append one measurement to the store.

    Args:
        uri: Baseline store URI.
        row: The measurement to record.
    """
    payload = json.dumps(row.as_dict(), sort_keys=True, separators=(",", ":"))
    _backend(uri).append_text(_key_for(row.contract_hash), payload + "\n")
