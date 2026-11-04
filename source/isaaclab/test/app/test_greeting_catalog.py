# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the remote greeting catalog and terminal-safe art payloads."""

import hashlib
import json
from datetime import date

import pytest

from isaaclab.app._greeting_catalog import AssetReference, decode_greeting, select_assets

pytestmark = pytest.mark.kitless

_TODAY = date(2026, 12, 24)
_NARROW = "\x1b[0;38;2;255;0;128;48;2;0;32;255m新春" + " " * 20 + "\x1b[39;49m"
_WIDE = "\x1b[38;2;0;255;0mHappy holidays" + " " * 60 + "\x1b[0m"


def _catalog() -> dict:
    return {
        "schema_version": 1,
        "active_collection": "default",
        "collections": {
            "default": {"asset_ids": ["daily"]},
            "holiday": {"asset_ids": ["first", "second"]},
        },
        "assets": {
            "second": {"path": "assets/second." + "b" * 16 + ".json", "sha256": "b" * 64, "bytes": 200},
            "first": {"path": "assets/first." + "a" * 16 + ".json", "sha256": "a" * 64, "bytes": 100},
            "daily": {"path": "assets/daily." + "c" * 16 + ".json", "sha256": "c" * 64, "bytes": 50},
        },
        "schedule": {
            "timezone": "UTC",
            "events": [
                {"id": "later", "collection": "holiday", "start": "2027-01-01", "end": "2027-01-02"},
                {"id": "winter", "collection": "holiday", "start": "2026-12-20", "end": "2026-12-27"},
            ],
        },
    }


def _greeting() -> dict:
    return {
        "schema_version": 1,
        "id": "first",
        "collection": "holiday",
        "slots": [
            {"cols": 24, "rows": 12, "frame_duration_ms": 50, "frames": ["\n".join([_NARROW] * 12)]},
            {"cols": 74, "rows": 12, "frame_duration_ms": 5000, "frames": ["\n".join([_WIDE] * 12)]},
        ],
    }


def _payload(greeting: object) -> tuple[bytes, AssetReference]:
    blob = json.dumps(greeting, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return blob, AssetReference("first", "holiday", f"assets/first.{digest[:16]}.json", digest, len(blob))


def test_utc_schedule_boundaries_order_and_default_fallback():
    manifest = _catalog()
    for day, identifiers in ((19, ()), (20, ("first", "second")), (27, ("first", "second")), (28, ())):
        assert tuple(ref.identifier for ref in select_assets(manifest, date(2026, 12, day))) == identifiers, day
    assert select_assets(manifest, _TODAY)[0] == AssetReference(
        "first", "holiday", "assets/first." + "a" * 16 + ".json", "a" * 64, 100
    )
    manifest["active_collection"] = "holiday"
    manifest["schedule"]["events"][1]["collection"] = "default"
    assert select_assets(manifest, _TODAY) == ()
    assert tuple(ref.identifier for ref in select_assets(manifest, date(2026, 1, 1))) == ("first", "second")
    del manifest["schedule"]
    assert tuple(ref.identifier for ref in select_assets(manifest, _TODAY)) == ("first", "second")


def test_rejects_invalid_or_overlapping_utc_events():
    for index, field, value in (
        (1, "start", "20261220"),
        (1, "end", "2026-12-19"),
        (1, "collection", "missing"),
        (1, "id", "later"),
        (0, "start", "2026-12-27"),
    ):
        manifest = _catalog()
        manifest["schedule"]["events"][index][field] = value
        with pytest.raises(ValueError):
            select_assets(manifest, _TODAY)
    manifest = _catalog()
    manifest["schedule"]["timezone"] = "Europe/Zurich"
    with pytest.raises(ValueError):
        select_assets(manifest, _TODAY)


def test_validates_catalog_references_and_unselected_asset_integrity():
    for field, value in (
        ("path", "assets/../daily.cccccccccccccccc.json"),
        ("path", "https://example.com/assets/daily.cccccccccccccccc.json"),
        ("sha256", "b" * 64),
        ("bytes", 262145),
    ):
        manifest = _catalog()
        manifest["assets"]["daily"][field] = value
        with pytest.raises(ValueError):
            select_assets(manifest, _TODAY)
    for identifiers in (["first", "first"], ["missing"]):
        manifest = _catalog()
        manifest["collections"]["holiday"]["asset_ids"] = identifiers
        with pytest.raises(ValueError):
            select_assets(manifest, _TODAY)
    manifest = _catalog()
    manifest["schema_version"] = 2
    with pytest.raises(ValueError):
        select_assets(manifest, _TODAY)


def test_decodes_first_frame_pair_with_truecolor_and_cjk_cell_dimensions():
    greeting = _greeting()
    greeting["slots"][0]["frames"].append("\n".join([" " * 24] * 12))
    assert decode_greeting(*_payload(greeting)) == ("\n".join([_NARROW] * 12), "\n".join([_WIDE] * 12))


def test_rejects_corrupt_malformed_or_oversized_payloads():
    blob, reference = _payload(_greeting())
    for invalid in (reference._replace(size=len(blob) + 1), reference._replace(sha256="0" * 64)):
        with pytest.raises(ValueError):
            decode_greeting(blob, invalid)
    for field, value in (("schema_version", 2), ("id", "second"), ("collection", "default"), ("padding", " " * 262144)):
        greeting = _greeting()
        greeting[field] = value
        with pytest.raises(ValueError):
            decode_greeting(*_payload(greeting))
    with pytest.raises(ValueError):
        decode_greeting(*_payload([]))


def test_rejects_invalid_slot_order_cell_dimensions_and_frame_bounds():
    greeting = _greeting()
    greeting["slots"].reverse()
    with pytest.raises(ValueError):
        decode_greeting(*_payload(greeting))
    for invalid in (
        {"frames": ["\n".join(["新" * 24] * 12)]},
        {"frames": ["\n".join([_NARROW] * 12)] * 25},
        {"frame_duration_ms": 49},
    ):
        greeting = _greeting()
        greeting["slots"][0].update(invalid)
        with pytest.raises(ValueError):
            decode_greeting(*_payload(greeting))


@pytest.mark.parametrize(
    "control", ["\x1b[2J", "\x1b]8;;x\x07", "\x1b[1m", "\x1b[38;2;256;0;0m", "\r", "\x9b", "\u2028", "\u2029"]
)
def test_rejects_terminal_controls_in_later_frames(control):
    greeting = _greeting()
    frame = greeting["slots"][1]["frames"][0].replace("Happy", "Happy" + control)
    greeting["slots"][1]["frames"].append(frame)
    with pytest.raises(ValueError):
        decode_greeting(*_payload(greeting))
