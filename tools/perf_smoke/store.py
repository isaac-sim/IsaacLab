# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Append-only baseline store, backed by one Azure Blob Storage container.

The store is the only module that performs I/O or touches credentials.

Configuration is a single environment variable, :data:`BLOB_URL_ENV`, holding the
container SAS URL. That URL already names the account and the container, and the layout
below is fixed in code, so there is nothing else to configure.

Layout::

    baselines/v1/<contract_hash>/<YYYY-MM>/<commit12>-<run_id>.json

One **immutable blob per measurement**, never an appended log. Three properties of the
container drive that choice:

* **Delete is not granted.** A single corrupt append-log would poison every read for its
  contract permanently. One blob per row confines corruption to one sample, which
  :func:`read` skips with a warning.
* **Overwrite is granted**, so clobbering is possible. Writing with ``overwrite=False``
  under a name derived from commit and run id makes it structurally impossible instead,
  and the name, derived from the commit and run id, makes a re-run of the same CI job
  record the sample once rather than once per attempt.
* **Only the key prefix is an index.** Blob storage is a flat key-to-blob map, and
  blob index tags needs SAS permissions this credential does not carry.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

from .metrics import METRICS, PerfSmokeError, mapping, number

BLOB_PREFIX = "baselines"
"""Root key prefix for stored measurements."""

LAYOUT_VERSION = "v1"
"""Storage layout generation."""

MAX_LOOKBACK_MONTHS = 3
"""How many monthly partitions :func:`read` will walk back before giving up."""

BLOB_URL_ENV = "ISAACLAB_BLOB_URL"
"""Environment variable holding the container SAS URL. Never an argument."""

EXPIRY_WARNING_DAYS = 7
"""Warn this far ahead of the SAS expiry, so the credential does not lapse silently."""

# These are the blob SDK's ExponentialRetry names, NOT azure-core's generic
# retry_backoff_factor/retry_backoff_max. The blob SDK substitutes its own retry policy,
# so the generic names are accepted and then silently discarded.
_RETRY_TOTAL = 3
_RETRY_INITIAL_BACKOFF_S = 2
_RETRY_INCREMENT_BASE_S = 2
_CONNECTION_TIMEOUT_S = 10
_READ_TIMEOUT_S = 30
_TOTAL_BUDGET_S = 120

# Dots are excluded; no way to spell a ".." segment.
_SAFE_NAME = re.compile(r"[^0-9a-zA-Z_-]+")
_QUERY = re.compile(r"\?[^\s\"'<>]*")
_METADATA_KEY = re.compile(r"[^0-9a-zA-Z_]+")


def _warn(message: str) -> None:
    """Emit a non-fatal warning that GitHub Actions surfaces as an annotation."""
    print(f"::warning::perf-smoke: {message}", file=sys.stderr)


def scrub(text: str) -> str:
    """Redact query strings, which is where a SAS token lives, from ``text``."""
    return _QUERY.sub("?<redacted>", text)


def is_configured() -> bool:
    """Return whether a credential is available for the store."""
    return bool(os.environ.get(BLOB_URL_ENV, "").strip())


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
        PerfSmokeError: If required fields are missing or malformed.
    """
    payload = mapping(data, source)
    stored_metrics = mapping(payload.get("metrics"), f"{source}.metrics")
    values: dict[str, float] = {}
    for metric in METRICS:
        # Rows written before a metric was added simply lack it.
        if metric.name not in stored_metrics:
            continue
        try:
            values[metric.name] = number(stored_metrics[metric.name], f"{source}.metrics.{metric.name}")
        except PerfSmokeError as exc:
            # Same treatment as an absent metric. A malformed advisory value must not
            # cost us the gating one recorded beside it, which cannot be re-recorded.
            _warn(f"ignoring metric in {source}: {exc}")
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


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _slug(value: str, fallback: str) -> str:
    """Reduce ``value`` to characters that are safe in a blob name."""
    cleaned = _SAFE_NAME.sub("-", value).strip("-")
    return cleaned or fallback


def month_of(timestamp: str) -> str:
    """Return the ``YYYY-MM`` partition for an ISO-8601 timestamp.

    Raises:
        PerfSmokeError: If ``timestamp`` does not start with a calendar month.
    """
    head = timestamp.strip()[:7]
    try:
        datetime.strptime(head, "%Y-%m")
    except ValueError:
        raise PerfSmokeError(f"timestamp {timestamp!r} must be ISO-8601, starting YYYY-MM") from None
    return head


def _months_back(now: datetime, count: int) -> list[str]:
    """Return ``count`` ``YYYY-MM`` partitions ending at ``now``, newest first."""
    months: list[str] = []
    year, month = now.year, now.month
    for _ in range(count):
        months.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return months


def _contract_prefix(contract_hash: str) -> str:
    """Return the key prefix holding every row for one contract."""
    return f"{BLOB_PREFIX}/{LAYOUT_VERSION}/{contract_hash}"


def _row_name(row: BaselineRow) -> str:
    """Return the blob name for ``row``.

    The name is a pure function of commit and run id, so re-running the same CI job
    reproduces it exactly and the ``overwrite=False`` write becomes idempotent.
    """
    commit = _slug(row.commit, "unknown")[:12].lower()
    run_id = _slug(row.run_id, "norun")
    return f"{_contract_prefix(row.contract_hash)}/{month_of(row.timestamp)}/{commit}-{run_id}.json"


def _row_metadata(row: BaselineRow) -> dict[str, str]:
    """Provenance visible in a listing without downloading. Never read back by code."""
    workload = row.contract.get("workload") if isinstance(row.contract, dict) else None
    runtime = row.contract.get("runtime") if isinstance(row.contract, dict) else None
    candidates = {
        "contract_hash": row.contract_hash,
        "commit": row.commit,
        "run_id": row.run_id,
        "uploaded_on": row.timestamp,
    }
    if isinstance(workload, dict):
        candidates["task"] = str(workload.get("task", ""))
        candidates["physics_backend"] = str(workload.get("physics_backend", ""))
        # None means the run did no rendering; str() would write the word "None".
        candidates["render_backend"] = str(workload.get("render_backend") or "")
    if isinstance(runtime, dict):
        candidates["gpu_model"] = str(runtime.get("gpu_model", ""))
    metadata: dict[str, str] = {}
    for key, value in candidates.items():
        # Azure metadata keys must be valid C# identifiers; values must be ASCII.
        safe_key = _METADATA_KEY.sub("_", key).lstrip("0123456789")
        safe_value = value.encode("ascii", "ignore").decode("ascii").strip()
        if safe_key and safe_value:
            metadata[safe_key] = safe_value[:256]
    return metadata


# ---------------------------------------------------------------------------
# Container access
# ---------------------------------------------------------------------------


def make_container_client(sas_url: str) -> Any:
    """Build an Azure container client from a container SAS URL.

    Isolated so the SDK construction, including the retry tuning above, has one call site.

    Raises:
        PerfSmokeError: If ``azure-storage-blob`` is not available.
    """
    try:
        from azure.storage.blob import ContainerClient
    except ImportError as exc:  # pragma: no cover - depends on runner image
        raise PerfSmokeError(
            "the baseline store requires azure-storage-blob (see tools/perf_smoke/requirements.txt)"
        ) from exc
    try:
        return ContainerClient.from_container_url(
            sas_url,
            retry_total=_RETRY_TOTAL,
            initial_backoff=_RETRY_INITIAL_BACKOFF_S,
            increment_base=_RETRY_INCREMENT_BASE_S,
            connection_timeout=_CONNECTION_TIMEOUT_S,
            read_timeout=_READ_TIMEOUT_S,
        )
    except ValueError as exc:
        raise PerfSmokeError(f"${BLOB_URL_ENV} is not a container SAS URL: {scrub(str(exc))}") from None


def sas_expiry(sas_url: str) -> datetime | None:
    """Return the SAS expiry from the URL's ``se`` parameter, or ``None`` if absent."""
    raw = (parse_qs(urlparse(sas_url).query).get("se") or [""])[0]
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _check_expiry(sas_url: str, now: datetime) -> None:
    """Fail on an expired credential, warn before it lapses."""
    expires = sas_expiry(sas_url)
    if expires is None:
        return
    if expires <= now:
        raise PerfSmokeError(
            f"the {BLOB_URL_ENV} credential expired on {expires.date().isoformat()}; it must be reissued"
        )
    remaining = expires - now
    if remaining < timedelta(days=EXPIRY_WARNING_DAYS):
        _warn(f"the {BLOB_URL_ENV} credential expires on {expires.date().isoformat()} ({remaining.days} days)")


def _describe(exc: Exception, action: str) -> str:
    """Render a storage failure without leaking the SAS token."""
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        code = getattr(exc, "error_code", None)
        detail = f" ({code})" if code else ""
        return (
            f"{action} was denied by the storage account (HTTP {status}){detail}; the"
            f" ${BLOB_URL_ENV} credential is expired, revoked, truncated, or lacks a needed permission"
        )
    return f"{action} failed: {scrub(str(exc))}"


class _Container:
    """The baseline container, addressed by the SAS URL in the environment."""

    def __init__(self) -> None:
        sas_url = os.environ.get(BLOB_URL_ENV, "").strip()
        if not sas_url:
            raise PerfSmokeError(f"the baseline store requires the container SAS URL in ${BLOB_URL_ENV}")
        _check_expiry(sas_url, datetime.now(timezone.utc))
        self._client = make_container_client(sas_url)

    def list_names(self, prefix: str) -> list[str]:
        """Return the keys under ``prefix``, oldest first."""
        from azure.core.exceptions import AzureError

        try:
            blobs = list(self._client.list_blobs(name_starts_with=prefix))
        except AzureError as exc:
            raise PerfSmokeError(_describe(exc, f"listing {prefix}")) from None
        # last_modified is the creation time here: rows are written once and never updated.
        blobs.sort(key=lambda blob: (blob.last_modified, blob.name))
        return [blob.name for blob in blobs]

    def read_text(self, name: str) -> str | None:
        """Return the blob's contents, or ``None`` when it does not exist."""
        from azure.core.exceptions import AzureError, ResourceNotFoundError

        try:
            return self._client.download_blob(name, encoding="utf-8").readall()
        except ResourceNotFoundError:
            return None
        except AzureError as exc:
            raise PerfSmokeError(_describe(exc, f"reading {name}")) from None

    def create_text(self, name: str, text: str, metadata: dict[str, str]) -> bool:
        """Create the blob. Return ``False`` if it already exists, never overwrite."""
        from azure.core.exceptions import AzureError, ResourceExistsError

        try:
            self._client.upload_blob(name=name, data=text.encode("utf-8"), overwrite=False, metadata=metadata)
        except ResourceExistsError:
            return False
        except AzureError as exc:
            raise PerfSmokeError(_describe(exc, f"writing {name}")) from None
        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read(contract_hash: str, limit: int, now: datetime | None = None) -> list[BaselineRow]:
    """Return the most recent rows recorded for ``contract_hash``.

    Walks monthly partitions backwards from ``now`` until ``limit`` rows are in hand or
    :data:`MAX_LOOKBACK_MONTHS` is exhausted, then fetches only the rows it will use.

    Args:
        contract_hash: Digest identifying poolable runs.
        limit: Maximum number of rows to return, newest last.
        now: Reference time for the month walk. Defaults to the current UTC time.

    Returns:
        Up to ``limit`` rows in chronological order. Empty when the contract has never
        been recorded, or when no sample falls inside the lookback window.

    Raises:
        PerfSmokeError: If the store cannot be reached.
    """
    if limit <= 0:
        raise PerfSmokeError("limit must be a positive integer")
    deadline = time.monotonic() + _TOTAL_BUDGET_S
    container = _Container()
    base = _contract_prefix(contract_hash)

    names: list[str] = []
    for month in _months_back(now or datetime.now(timezone.utc), MAX_LOOKBACK_MONTHS):
        if time.monotonic() > deadline:
            raise PerfSmokeError(f"the baseline store did not respond within {_TOTAL_BUDGET_S}s")
        names = container.list_names(f"{base}/{month}/") + names
        if len(names) >= limit:
            break

    rows: list[BaselineRow] = []
    for name in names[-limit:]:
        if time.monotonic() > deadline:
            raise PerfSmokeError(f"the baseline store did not respond within {_TOTAL_BUDGET_S}s")
        try:
            text = container.read_text(name)
            if text is None:
                continue
            rows.append(parse_row(json.loads(text), name))
        except (PerfSmokeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            _warn(f"skipping unreadable baseline row {name}: {exc}")
    return rows


def write(row: BaselineRow) -> bool:
    """Record one measurement, without ever overwriting an existing one.

    Args:
        row: The measurement to record.

    Returns:
        ``True`` when the row was created, ``False`` when an identical name exits.

    Raises:
        PerfSmokeError: If the store cannot be reached or the row cannot be named.
    """
    payload = json.dumps(row.as_dict(), sort_keys=True, separators=(",", ":"))
    return _Container().create_text(_row_name(row), payload, _row_metadata(row))
