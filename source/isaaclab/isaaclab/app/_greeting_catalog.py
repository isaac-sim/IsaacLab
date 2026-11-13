# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the remote greeting protocol without performing I/O or importing a simulator."""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import unicodedata
from typing import NamedTuple

from rich.cells import cell_len

_MAX_ASSET_BYTES = 262144
_ASSET_PATH = re.compile(r"assets/[a-z0-9-]+\.([0-9a-f]{16})\.json")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_SGR = re.compile(r"\x1b\[([0-9;]+)m")


def select_assets(manifest: object, on_date: datetime.date) -> tuple[AssetReference, ...]:
    """Validate a catalog and select its scheduled or fallback collection.

    Schedule dates are inclusive UTC calendar dates. The built-in default collection
    requires no remote assets, so selecting it returns an empty tuple.

    Raises:
        ValueError: If the catalog or schedule violates the greeting protocol.
    """
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int:
        raise ValueError("A greeting catalog must declare schema version 1.")
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported greeting catalog schema version.")
    collections = manifest.get("collections")
    assets = manifest.get("assets")
    if not isinstance(collections, dict) or not isinstance(assets, dict):
        raise ValueError("A greeting catalog must contain collections and assets.")

    for identifier, asset in assets.items():
        if not isinstance(identifier, str) or not identifier or not isinstance(asset, dict):
            raise ValueError("Greeting asset entries must have nonempty identifiers and metadata.")
        path = asset.get("path")
        digest = asset.get("sha256")
        size = asset.get("bytes")
        path_match = _ASSET_PATH.fullmatch(path) if isinstance(path, str) else None
        if (
            path_match is None
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or path_match[1] != digest[:16]
            or type(size) is not int
            or not 1 <= size <= _MAX_ASSET_BYTES
        ):
            raise ValueError("Greeting asset paths, hashes, and byte counts must be valid and bounded.")

    references: dict[str, tuple[AssetReference, ...]] = {}
    seen_assets: set[str] = set()
    for collection, metadata in collections.items():
        if not isinstance(collection, str) or not collection or not isinstance(metadata, dict):
            raise ValueError("Greeting collections must have nonempty names and metadata.")
        identifiers = metadata.get("asset_ids")
        if not isinstance(identifiers, list):
            raise ValueError("A greeting collection must list its asset identifiers.")
        collection_references = []
        for identifier in identifiers:
            if not isinstance(identifier, str) or identifier not in assets or identifier in seen_assets:
                raise ValueError("Greeting asset identifiers must exist and belong to one collection.")
            seen_assets.add(identifier)
            asset = assets[identifier]
            collection_references.append(
                AssetReference(identifier, collection, asset["path"], asset["sha256"], asset["bytes"])
            )
        references[collection] = tuple(collection_references)

    selected = manifest.get("active_collection")
    if not isinstance(selected, str) or selected not in references:
        raise ValueError("The active greeting collection must exist.")
    if "schedule" in manifest:
        schedule = manifest["schedule"]
        if not isinstance(schedule, dict) or schedule.get("timezone") != "UTC":
            raise ValueError("Greeting schedules must use UTC.")
        events = schedule.get("events")
        if not isinstance(events, list):
            raise ValueError("A greeting schedule must list its events.")
        intervals = []
        seen_events: set[str] = set()
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("Greeting schedule events must contain metadata.")
            identifier = event.get("id")
            collection = event.get("collection")
            if not isinstance(identifier, str) or not identifier or identifier in seen_events:
                raise ValueError("Greeting schedule event identifiers must be nonempty and unique.")
            if not isinstance(collection, str) or collection not in references:
                raise ValueError("Scheduled greeting collections must exist.")
            seen_events.add(identifier)
            start = _parse_date(event.get("start"))
            end = _parse_date(event.get("end"))
            if start > end:
                raise ValueError("Greeting schedule events must end on or after their start.")
            intervals.append((start, end, collection))
        previous_end = None
        for start, end, collection in sorted(intervals):
            if previous_end is not None and start <= previous_end:
                raise ValueError("Greeting schedule events must not overlap.")
            previous_end = end
            if start <= on_date <= end:
                selected = collection
    return () if selected == "default" else references[selected]


def decode_greeting(blob: bytes, reference: AssetReference) -> tuple[str, str]:
    """Check an immutable greeting payload and return its narrow and wide logos.

    Each slot contains one static frame. Only terminal-safe reset and truecolor SGR
    sequences are accepted within the fixed cell dimensions.

    Raises:
        ValueError: If integrity, schema, dimensions, or terminal safety checks fail.
    """
    if (
        not isinstance(blob, bytes)
        or not 1 <= len(blob) <= _MAX_ASSET_BYTES
        or type(reference.size) is not int
        or len(blob) != reference.size
        or hashlib.sha256(blob).hexdigest() != reference.sha256
    ):
        raise ValueError("Greeting payload byte count or SHA-256 digest does not match.")
    try:
        greeting = json.loads(blob.decode("utf-8"))
    except RecursionError as error:
        raise ValueError("Greeting JSON is too deeply nested.") from error
    if (
        not isinstance(greeting, dict)
        or type(greeting.get("schema_version")) is not int
        or greeting["schema_version"] != 1
        or greeting.get("id") != reference.identifier
        or greeting.get("collection") != reference.collection
    ):
        raise ValueError("Greeting payload schema or identity does not match its catalog reference.")
    slots = greeting.get("slots")
    if not isinstance(slots, list) or len(slots) != 2:
        raise ValueError("A greeting must contain its narrow and wide slots in order.")
    logos = []
    for slot, cols in zip(slots, (24, 74)):
        if (
            not isinstance(slot, dict)
            or slot.keys() != {"cols", "rows", "frame"}
            or type(slot.get("cols")) is not int
            or slot["cols"] != cols
            or type(slot.get("rows")) is not int
            or slot["rows"] != 12
        ):
            raise ValueError("Greeting slots must contain one frame, ordered 24 by 12 and 74 by 12 cells.")
        frame = slot["frame"]
        _validate_frame(frame, cols)
        logos.append(frame)
    return logos[0], logos[1]


class AssetReference(NamedTuple):
    """Immutable catalog metadata for one greeting asset."""

    identifier: str
    collection: str
    path: str
    sha256: str
    size: int


def _parse_date(value: object) -> datetime.date:
    """Parse only the protocol's extended ISO calendar date format."""
    if not isinstance(value, str) or _DATE.fullmatch(value) is None:
        raise ValueError("Greeting schedule dates must use YYYY-MM-DD.")
    return datetime.date.fromisoformat(value)


def _validate_frame(frame: object, cols: int) -> None:
    """Reject terminal commands and measure plain text using terminal cell widths."""
    if not isinstance(frame, str):
        raise ValueError("Greeting frames must be strings.")
    segments = []
    position = 0
    for match in _SGR.finditer(frame):
        segments.append(frame[position : match.start()])
        parameters = [int(value) for value in match[1].split(";")]
        index = 0
        while index < len(parameters):
            code = parameters[index]
            if code in (0, 39, 49):
                index += 1
            elif (
                code in (38, 48)
                and index + 4 < len(parameters)
                and parameters[index + 1] == 2
                and all(0 <= channel <= 255 for channel in parameters[index + 2 : index + 5])
            ):
                index += 5
            else:
                raise ValueError("Greeting frames may contain only reset and truecolor SGR sequences.")
        position = match.end()
    segments.append(frame[position:])
    plain = "".join(segments)
    if any(
        character != "\n" and unicodedata.category(character) in ("Cc", "Cf", "Cs", "Zl", "Zp") for character in plain
    ):
        raise ValueError("Greeting frames must not contain terminal or Unicode controls.")
    lines = plain.split("\n")
    if len(lines) != 12 or any(cell_len(line) != cols for line in lines):
        raise ValueError("Greeting frame dimensions must match their slot in terminal cells.")
