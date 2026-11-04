# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the remote greeting catalog and terminal-safe art payloads."""

import datetime
import hashlib
import json

import pytest

from isaaclab.app._greeting_catalog import AssetReference, decode_greeting, select_assets

pytestmark = pytest.mark.kitless

_TODAY = datetime.date(2026, 12, 24)
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


@pytest.mark.parametrize(
    ("on_date", "identifiers"),
    [
        (datetime.date(2026, 12, 19), ()),
        (datetime.date(2026, 12, 20), ("first", "second")),
        (datetime.date(2026, 12, 27), ("first", "second")),
        (datetime.date(2026, 12, 28), ()),
    ],
)
def test_schedule_is_inclusive_and_preserves_collection_order(on_date, identifiers):
    references = select_assets(_catalog(), on_date)

    assert tuple(reference.identifier for reference in references) == identifiers
    if references:
        assert references[0] == AssetReference("first", "holiday", "assets/first." + "a" * 16 + ".json", "a" * 64, 100)


def test_active_collection_is_fallback_and_scheduled_default_never_fetches():
    manifest = _catalog()
    manifest["active_collection"] = "holiday"
    manifest["schedule"]["events"][1]["collection"] = "default"

    assert select_assets(manifest, _TODAY) == ()
    assert tuple(reference.identifier for reference in select_assets(manifest, datetime.date(2026, 1, 1))) == (
        "first",
        "second",
    )
    del manifest["schedule"]
    assert tuple(reference.identifier for reference in select_assets(manifest, _TODAY)) == ("first", "second")


@pytest.mark.parametrize(
    ("field", "value"),
    [("start", "20261220"), ("end", "2026-12-19"), ("collection", "missing"), ("id", "later")],
)
def test_rejects_invalid_schedule_events(field, value):
    manifest = _catalog()
    manifest["schedule"]["events"][1][field] = value

    with pytest.raises(ValueError):
        select_assets(manifest, _TODAY)


def test_rejects_overlapping_unsorted_events_and_non_utc_schedule():
    manifest = _catalog()
    manifest["schedule"]["events"][0]["start"] = "2026-12-27"
    with pytest.raises(ValueError):
        select_assets(manifest, _TODAY)

    manifest = _catalog()
    manifest["schedule"]["timezone"] = "Europe/Zurich"
    with pytest.raises(ValueError):
        select_assets(manifest, _TODAY)


@pytest.mark.parametrize(
    "path",
    ["../first.aaaaaaaaaaaaaaaa.json", "assets/../first.aaaaaaaaaaaaaaaa.json", "https://example.com/a.json"],
)
def test_rejects_unsafe_paths_even_for_unselected_assets(path):
    manifest = _catalog()
    manifest["assets"]["daily"]["path"] = path

    with pytest.raises(ValueError):
        select_assets(manifest, _TODAY)


@pytest.mark.parametrize(("field", "value"), [("sha256", "b" * 64), ("bytes", 262145), ("bytes", True)])
def test_rejects_invalid_asset_integrity_metadata(field, value):
    manifest = _catalog()
    manifest["assets"]["daily"][field] = value

    with pytest.raises(ValueError):
        select_assets(manifest, _TODAY)


def test_rejects_duplicate_and_unknown_collection_asset_ids():
    manifest = _catalog()
    manifest["collections"]["holiday"]["asset_ids"] = ["first", "first"]
    with pytest.raises(ValueError):
        select_assets(manifest, _TODAY)

    manifest["collections"]["holiday"]["asset_ids"] = ["missing"]
    with pytest.raises(ValueError):
        select_assets(manifest, _TODAY)


@pytest.mark.parametrize("manifest", [None, {"schema_version": True}, {"schema_version": 2}])
def test_rejects_invalid_manifest_schema(manifest):
    with pytest.raises(ValueError):
        select_assets(manifest, _TODAY)


def test_decodes_first_frames_with_truecolor_and_cjk_cell_dimensions():
    greeting = _greeting()
    greeting["slots"][0]["frames"].append("\n".join([" " * 24] * 12))
    blob, reference = _payload(greeting)

    assert decode_greeting(blob, reference) == ("\n".join([_NARROW] * 12), "\n".join([_WIDE] * 12))


def test_checks_payload_size_and_digest_before_decoding():
    blob, reference = _payload(_greeting())
    with pytest.raises(ValueError):
        decode_greeting(blob, reference._replace(size=len(blob) + 1))
    with pytest.raises(ValueError):
        decode_greeting(blob, reference._replace(sha256="0" * 64))

    oversized = b" " * 262145
    digest = hashlib.sha256(oversized).hexdigest()
    with pytest.raises(ValueError):
        decode_greeting(oversized, reference._replace(size=len(oversized), sha256=digest))


@pytest.mark.parametrize(("field", "value"), [("schema_version", 2), ("id", "second"), ("collection", "default")])
def test_rejects_payload_schema_or_identity_mismatch(field, value):
    greeting = _greeting()
    greeting[field] = value
    blob, reference = _payload(greeting)

    with pytest.raises(ValueError):
        decode_greeting(blob, reference)


@pytest.mark.parametrize("blob", [b"{", b"\xff", b"[]"])
def test_malformed_json_is_a_value_error(blob):
    digest = hashlib.sha256(blob).hexdigest()
    reference = AssetReference("first", "holiday", f"assets/first.{digest[:16]}.json", digest, len(blob))

    with pytest.raises(ValueError):
        decode_greeting(blob, reference)


def test_rejects_reordered_slots_and_wrong_cell_dimensions():
    greeting = _greeting()
    greeting["slots"].reverse()
    blob, reference = _payload(greeting)
    with pytest.raises(ValueError):
        decode_greeting(blob, reference)

    greeting = _greeting()
    greeting["slots"][0]["frames"] = ["\n".join(["新" * 24] * 12)]
    blob, reference = _payload(greeting)
    with pytest.raises(ValueError):
        decode_greeting(blob, reference)


@pytest.mark.parametrize(
    "control",
    [
        "\x1b[2J",
        "\x1b]8;;https://example.com\x07",
        "\x1b[1m",
        "\x1b[38;2;256;0;0m",
        "\r",
        "\x9b",
        "\u2028",
        "\u2029",
    ],
)
def test_rejects_terminal_controls_in_later_frames(control):
    greeting = _greeting()
    frame = greeting["slots"][1]["frames"][0].replace("Happy", "Happy" + control)
    greeting["slots"][1]["frames"].append(frame)
    blob, reference = _payload(greeting)

    with pytest.raises(ValueError):
        decode_greeting(blob, reference)


def test_bounds_frame_count_and_duration():
    greeting = _greeting()
    greeting["slots"][0]["frames"] *= 25
    blob, reference = _payload(greeting)
    with pytest.raises(ValueError):
        decode_greeting(blob, reference)

    greeting = _greeting()
    greeting["slots"][0]["frame_duration_ms"] = 49
    blob, reference = _payload(greeting)
    with pytest.raises(ValueError):
        decode_greeting(blob, reference)
