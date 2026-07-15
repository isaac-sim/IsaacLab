# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-agnostic utilities for generating and persisting reset-state datasets."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any, TypeVar

import torch

_BatchT = TypeVar("_BatchT")


class _HashWriter:
    """Adapt a hashlib digest to the byte-writer interface used by the encoder."""

    def __init__(self) -> None:
        self.digest = hashlib.sha256()

    def write(self, value: bytes) -> None:
        """Append bytes to the digest."""
        self.digest.update(value)


def _write_bytes(sink: Any, value: bytes) -> None:
    """Write one length-delimited byte string to a hash or byte buffer."""
    sink.write(len(value).to_bytes(8, byteorder="big", signed=False))
    sink.write(value)


def _write_value(sink: Any, value: Any) -> None:
    """Write the canonical representation of one supported value."""
    if value is None:
        sink.write(b"none")
    elif isinstance(value, bool):
        sink.write(b"bool")
        sink.write(b"1" if value else b"0")
    elif isinstance(value, int):
        sink.write(b"int")
        _write_bytes(sink, str(value).encode("ascii"))
    elif isinstance(value, float):
        sink.write(b"float")
        _write_bytes(sink, value.hex().encode("ascii"))
    elif isinstance(value, str):
        sink.write(b"str")
        _write_bytes(sink, value.encode("utf-8"))
    elif isinstance(value, bytes):
        sink.write(b"bytes")
        _write_bytes(sink, value)
    elif isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError("Reset-dataset hashes support only dense, strided tensors.")
        tensor = value.detach().cpu().contiguous()
        sink.write(b"tensor")
        _write_bytes(sink, str(tensor.dtype).encode("ascii"))
        _write_value(sink, tuple(tensor.shape))
        raw_bytes = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        _write_bytes(sink, raw_bytes)
    elif isinstance(value, Mapping):
        sink.write(b"mapping")
        encoded_items: list[tuple[bytes, Any]] = []
        for key, item in value.items():
            key_buffer = BytesIO()
            _write_value(key_buffer, key)
            encoded_items.append((key_buffer.getvalue(), item))
        encoded_items.sort(key=lambda pair: pair[0])
        _write_value(sink, len(encoded_items))
        for encoded_key, item in encoded_items:
            _write_bytes(sink, encoded_key)
            _write_value(sink, item)
    elif isinstance(value, tuple):
        sink.write(b"tuple")
        _write_value(sink, len(value))
        for item in value:
            _write_value(sink, item)
    elif isinstance(value, list):
        sink.write(b"list")
        _write_value(sink, len(value))
        for item in value:
            _write_value(sink, item)
    else:
        raise TypeError(f"Unsupported reset-dataset hash value type: {type(value).__name__}.")


def reset_dataset_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for nested primitive and tensor data.

    Mapping insertion order and tensor device do not affect the result. Concrete sequence types,
    tensor dtypes, and tensor shapes remain part of the digest.

    Args:
        value: Nested mappings, lists, tuples, primitive values, or dense tensors.

    Returns:
        The lowercase hexadecimal SHA-256 digest.

    Raises:
        TypeError: If the value contains an unsupported type or tensor layout.
    """
    writer = _HashWriter()
    _write_value(writer, value)
    return writer.digest.hexdigest()


def reset_dataset_content_digest(
    payload: Mapping[str, Any],
    *,
    digest_key: str = "content_sha256",
) -> str:
    """Return a dataset payload digest while excluding its own digest field.

    Args:
        payload: Dataset payload to hash.
        digest_key: Top-level field that stores the resulting content digest.

    Returns:
        The content SHA-256 digest.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("Reset dataset payload must be a mapping.")
    if not isinstance(digest_key, str) or not digest_key:
        raise ValueError("digest_key must be a non-empty string.")
    return reset_dataset_digest({key: value for key, value in payload.items() if key != digest_key})


def reset_dataset_validate_header(
    payload: Mapping[str, Any],
    *,
    expected_format: str,
    expected_schema_version: int,
    expected_contract: Any,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate the common envelope and integrity hashes of a reset dataset.

    Task-specific validators remain responsible for state keys, tensor shapes, physical
    invariants, and metadata semantics.

    Args:
        payload: Dataset payload to validate.
        expected_format: Exact dataset format identifier.
        expected_schema_version: Exact schema version supported by the caller.
        expected_contract: Generator and task contract represented by ``contract_sha256``.

    Returns:
        The validated metadata and states mappings.

    Raises:
        TypeError: If the payload, metadata, or states are not mappings.
        ValueError: If a header field or integrity digest does not match.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("Reset dataset payload must be a mapping.")
    if not isinstance(expected_format, str) or not expected_format:
        raise ValueError("expected_format must be a non-empty string.")
    if not isinstance(expected_schema_version, int) or isinstance(expected_schema_version, bool):
        raise TypeError("expected_schema_version must be an integer.")
    if payload.get("format") != expected_format:
        raise ValueError(f"Expected reset dataset format {expected_format!r}.")
    if payload.get("schema_version") != expected_schema_version:
        raise ValueError(f"Expected reset dataset schema version {expected_schema_version}.")

    metadata = payload.get("metadata")
    states = payload.get("states")
    if not isinstance(metadata, Mapping):
        raise TypeError("Reset dataset metadata must be a mapping.")
    if not isinstance(states, Mapping):
        raise TypeError("Reset dataset states must be a mapping.")

    expected_contract_digest = reset_dataset_digest(expected_contract)
    if payload.get("contract_sha256") != expected_contract_digest:
        raise ValueError("Reset dataset contract digest does not match the expected contract.")
    if payload.get("content_sha256") != reset_dataset_content_digest(payload):
        raise ValueError("Reset dataset content digest does not match its payload.")
    return metadata, states


def reset_dataset_save_atomic(
    payload: Mapping[str, Any],
    output_path: str | Path,
    *,
    validator: Callable[[Mapping[str, Any]], None],
) -> Path:
    """Validate and atomically save a reset dataset with :func:`torch.save`.

    Args:
        payload: Complete dataset payload.
        output_path: Destination file path.
        validator: Dataset-specific validator called before the destination is modified.

    Returns:
        The resolved destination path.
    """
    if not callable(validator):
        raise TypeError("validator must be callable.")
    validator(payload)
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def reset_dataset_collect_batches(
    target_count: int,
    *,
    batch_size: int,
    max_candidate_count: int,
    evaluate_batch: Callable[[range], _BatchT],
    batch_count: Callable[[_BatchT], int],
    batch_slice: Callable[[_BatchT, int], _BatchT],
) -> tuple[list[_BatchT], int]:
    """Collect exactly the requested accepted samples from rejection-sampled batches.

    ``evaluate_batch`` receives a unique, contiguous range of candidate IDs. It owns proposal
    generation and validation, and returns only accepted samples. Keeping batch representation
    behind callbacks lets tasks retain efficient tensor dictionaries or custom batch types.

    Args:
        target_count: Number of accepted samples to collect.
        batch_size: Maximum number of candidates evaluated per callback.
        max_candidate_count: Maximum candidates evaluated before failing.
        evaluate_batch: Callback that proposes and validates one candidate-ID range.
        batch_count: Callback returning the accepted sample count in a batch.
        batch_slice: Callback retaining the first requested samples of a batch.

    Returns:
        A list of accepted batches and the number of candidates evaluated.

    Raises:
        ValueError: If counts are invalid or a callback reports an impossible batch size.
        RuntimeError: If the candidate budget is exhausted before reaching ``target_count``.
    """
    for name, value in (
        ("target_count", target_count),
        ("batch_size", batch_size),
        ("max_candidate_count", max_candidate_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")
    if target_count < 0:
        raise ValueError("target_count must be nonnegative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_candidate_count < target_count:
        raise ValueError("max_candidate_count cannot be smaller than target_count.")
    if target_count == 0:
        return [], 0

    batches: list[_BatchT] = []
    accepted_total = 0
    evaluated_total = 0
    while accepted_total < target_count and evaluated_total < max_candidate_count:
        candidate_count = min(batch_size, max_candidate_count - evaluated_total)
        candidate_ids = range(evaluated_total, evaluated_total + candidate_count)
        batch = evaluate_batch(candidate_ids)
        accepted_count = batch_count(batch)
        if not isinstance(accepted_count, int) or isinstance(accepted_count, bool):
            raise TypeError("batch_count must return an integer.")
        if not 0 <= accepted_count <= candidate_count:
            raise ValueError(
                f"Accepted batch count {accepted_count} is outside the candidate range [0, {candidate_count}]."
            )
        evaluated_total += candidate_count
        if accepted_count == 0:
            continue
        remaining = target_count - accepted_total
        if accepted_count > remaining:
            batch = batch_slice(batch, remaining)
            accepted_count = batch_count(batch)
            if accepted_count != remaining:
                raise ValueError("batch_slice did not return the requested accepted sample count.")
        batches.append(batch)
        accepted_total += accepted_count

    if accepted_total != target_count:
        raise RuntimeError(
            f"Reset-dataset rejection sampling accepted {accepted_total}/{target_count} samples "
            f"after evaluating {evaluated_total} candidates."
        )
    return batches, evaluated_total
