# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hash-chained provenance logger for governance decisions."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class ProvenanceLogger:
    """Append-only, hash-chained audit log for governance decisions.

    Every governance decision (ALLOW, CLAMP, DENY) is recorded as a
    JSON line with a cryptographic hash chain linking each record to
    its predecessor. This provides tamper-evident provenance that
    certifiers can independently verify.

    The hash chain uses SHA-256: each record's ``hash_curr`` is computed
    over the canonical JSON of the record (including ``hash_prev``),
    creating an append-only ledger.

    Attributes:
        records: In-memory list of all provenance records.
        chain_hash: Current head of the hash chain.
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        """Initialize the provenance logger.

        Args:
            log_path: Optional file path for persistent logging. If provided,
                records are appended as JSON lines. If ``None``, records are
                kept in memory only.
        """
        self.records: list[dict[str, Any]] = []
        self.chain_hash: str = "0" * 64  # genesis hash
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        step: int,
        agent_id: str | int,
        decision: str,
        proposed_action: list[float],
        applied_action: list[float],
        violations: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a governance decision.

        Args:
            step: Environment step number.
            agent_id: Identifier of the agent whose action was governed.
            decision: Governance verdict — one of ``"ALLOW"``, ``"CLAMP"``,
                or ``"DENY"``.
            proposed_action: The action the agent originally requested.
            applied_action: The action actually applied after governance.
            violations: List of constraint names that were violated (empty
                for ``"ALLOW"`` decisions).
            metadata: Optional additional context for this record.

        Returns:
            The complete provenance record including hash chain fields.
        """
        record = {
            "epoch": len(self.records),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": step,
            "agent_id": str(agent_id),
            "decision": decision,
            "proposed_action": proposed_action,
            "applied_action": applied_action,
            "violations": violations,
            "hash_prev": self.chain_hash,
        }
        if metadata:
            record["metadata"] = metadata

        # Compute hash_curr over canonical JSON (sorted keys, compact separators)
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
        record["hash_curr"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        self.chain_hash = record["hash_curr"]
        self.records.append(record)

        # Persist if log path configured
        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

        return record

    def verify_chain(self) -> tuple[bool, int]:
        """Verify the integrity of the hash chain.

        Returns:
            A tuple of ``(is_valid, last_valid_epoch)``. If the chain is
            intact, returns ``(True, len(records) - 1)``. If tampered,
            returns ``(False, epoch)`` where ``epoch`` is the first
            record that fails verification.
        """
        prev_hash = "0" * 64
        for i, record in enumerate(self.records):
            if record.get("hash_prev") != prev_hash:
                return False, i
            # Recompute hash_curr
            check = {k: v for k, v in record.items() if k != "hash_curr"}
            canonical = json.dumps(check, sort_keys=True, separators=(",", ":"))
            expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if record.get("hash_curr") != expected:
                return False, i
            prev_hash = record["hash_curr"]
        return True, len(self.records) - 1

    def summary(self) -> dict[str, int]:
        """Return summary statistics of governance decisions.

        Returns:
            Dictionary with counts of ALLOW, CLAMP, and DENY decisions.
        """
        counts = {"ALLOW": 0, "CLAMP": 0, "DENY": 0}
        for record in self.records:
            decision = record.get("decision", "")
            if decision in counts:
                counts[decision] += 1
        return counts
