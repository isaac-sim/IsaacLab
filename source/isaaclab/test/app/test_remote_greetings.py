# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for scheduled greetings, validated artwork, and offline startup."""

import hashlib
import io
import json
import os
import threading
import time
import urllib.request
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

import isaaclab.app._remote_greetings as remote
from isaaclab.app._greeting_catalog import AssetReference, decode_greeting, select_assets

pytestmark = pytest.mark.kitless


def _encode(payload, identifier="first"):
    blob = json.dumps(payload, ensure_ascii=False).encode()
    digest = hashlib.sha256(blob).hexdigest()
    return blob, AssetReference(identifier, "holiday", f"assets/{identifier}.{digest[:16]}.json", digest, len(blob))


def _add_asset(source, identifier, label="Season"):
    logos = tuple(
        "\n".join(["\x1b[0;38;2;255;0;128;48;2;0;32;255m" + text + "\x1b[39;49m"] * 12)
        for text in ("新春" + " " * 20, label.ljust(74))
    )
    payload = {
        "schema_version": 1,
        "id": identifier,
        "collection": "holiday",
        "slots": [{"cols": cols, "rows": 12, "frame": logo} for cols, logo in zip((24, 74), logos)],
    }
    blob, reference = _encode(payload, identifier)
    source.blobs[reference.path] = blob
    source.manifest["assets"][identifier] = {"path": reference.path, "sha256": reference.sha256, "bytes": len(blob)}
    source.manifest["collections"]["holiday"]["asset_ids"].append(identifier)
    return SimpleNamespace(logos=logos, payload=payload, blob=blob, reference=reference)


@pytest.fixture
def source(tmp_path, monkeypatch):
    source = SimpleNamespace(cache=tmp_path / "cache", requests=[], blobs={}, online=True)
    source.now = datetime(2026, 12, 24, tzinfo=timezone.utc)
    source.entered, source.release, source.done = (threading.Event() for _ in range(3))
    source.manifest = {
        "schema_version": 1,
        "active_collection": "default",
        "collections": {"default": {"asset_ids": []}, "holiday": {"asset_ids": []}},
        "assets": {},
        "schedule": {
            "timezone": "UTC",
            "events": [
                {"id": "later", "collection": "holiday", "start": "2027-01-01", "end": "2027-01-02"},
                {"id": "winter", "collection": "holiday", "start": "2026-12-20", "end": "2026-12-27"},
            ],
        },
    }
    source.first = _add_asset(source, "first")
    source.release.set()
    refresh = remote.RemoteGreetings._refresh

    def observed_refresh(loader):
        try:
            refresh(loader)
        finally:
            source.done.set()

    def open_url(opener, request, *args, **kwargs):
        path = request.full_url.removeprefix("https://greetings.example/IsaacLab/Greetings/")
        source.requests.append(path)
        source.entered.set()
        if not source.release.wait(2):
            raise TimeoutError("Test did not release its HTTP response.")
        if not source.online:
            raise OSError("Offline")
        return io.BytesIO(json.dumps(source.manifest).encode() if path == "manifest.json" else source.blobs[path])

    monkeypatch.setenv("XDG_CACHE_HOME", str(source.cache))
    monkeypatch.setattr(remote, "ISAACLAB_NUCLEUS_DIR", "https://greetings.example/IsaacLab")
    monkeypatch.setattr(remote, "datetime", SimpleNamespace(now=lambda tz: source.now))
    monkeypatch.setattr(remote.RemoteGreetings, "_refresh", observed_refresh)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", open_url)
    return source


def _load(source, **kwargs):
    source.done.clear()
    loader = remote.RemoteGreetings(**kwargs)
    assert source.done.wait(2), "The background refresh did not finish."
    return loader


def test_catalog_validation_and_utc_calendar(source, monkeypatch):
    catalog = source.manifest
    asset = catalog["assets"]["first"]
    event = catalog["schedule"]["events"][1]
    for entry, key, value in (
        (asset, "path", "assets/../first." + source.first.reference.sha256[:16] + ".json"),
        (asset, "path", "https://example.com/" + source.first.reference.path),
        (asset, "sha256", "0" * 64),
        (asset, "bytes", 262145),
        (event, "start", "20261220"),
        (event, "end", "2026-12-19"),
        (event, "id", "later"),
        (event, "collection", "missing"),
        (event, "end", "2027-01-01"),
        (catalog, "schema_version", 2),
        (catalog["schedule"], "timezone", "local"),
        (catalog["collections"]["holiday"], "asset_ids", ["first", "first"]),
        (catalog["collections"]["holiday"], "asset_ids", ["missing"]),
    ):
        with monkeypatch.context() as patch, pytest.raises(ValueError):
            patch.setitem(entry, key, value)
            select_assets(catalog, date(2026, 12, 19))
    _add_asset(source, "second")
    catalog["assets"] = dict(reversed(catalog["assets"].items()))
    for day, identifiers in ((19, ()), (20, ("first", "second")), (27, ("first", "second")), (28, ())):
        assert tuple(ref.identifier for ref in select_assets(catalog, date(2026, 12, day))) == identifiers, day
    catalog["active_collection"] = "holiday"
    event["collection"] = "default"
    assert select_assets(catalog, source.now.date()) == ()
    assert len(select_assets(catalog, date(2026, 1, 1))) == 2
    del catalog["schedule"]
    assert len(select_assets(catalog, source.now.date())) == 2


def test_payload_integrity_and_static_slots(source, monkeypatch):
    blob, reference = source.first.blob, source.first.reference
    assert decode_greeting(blob, reference) == source.first.logos
    for invalid in (reference._replace(size=len(blob) + 1), reference._replace(sha256="0" * 64)):
        with pytest.raises(ValueError):
            decode_greeting(blob, invalid)
    payload = source.first.payload
    slot = payload["slots"][0]
    for entry, key, value in (
        (payload, "schema_version", 2),
        (payload, "id", "wrong"),
        (payload, "collection", "default"),
        (payload, "padding", " " * 262144),
        (payload, "slots", payload["slots"][::-1]),
        (slot, "frame", "\n".join(["新" * 24] * 12)),
        (slot, "frame", [source.first.logos[0]]),
        (slot, "frames", [source.first.logos[0]]),
    ):
        with monkeypatch.context() as patch, pytest.raises(ValueError):
            patch.setitem(entry, key, value)
            decode_greeting(*_encode(payload))
    with pytest.raises(ValueError):
        decode_greeting(*_encode([]))
    controls = ["\x1b[2J", "\x1b]8;;x\x07", "\x1b[1m", "\x1b[38;2;256;0;0m", "\r", "\x9b", "\u2028", "\u2029"]
    for control in controls:
        payload["slots"][1]["frame"] = "\n".join([control + " " * 74] * 12)
        with pytest.raises(ValueError):
            decode_greeting(*_encode(payload))


def test_nonblocking_refresh_cache_variety_and_offline_fallback(source, monkeypatch):
    corrupt = source.first.blob.replace(b"Season", b"Tamper")
    source.blobs[source.first.reference.path] = corrupt
    assert _load(source).logos is None
    assert all(path.read_bytes() != corrupt for path in source.cache.rglob("*") if path.is_file())
    source.blobs[source.first.reference.path] = source.first.blob
    second = _add_asset(source, "second", "Spring")
    monkeypatch.setattr(
        remote, "random", SimpleNamespace(choice=lambda items: items[0], shuffle=lambda items: items.reverse())
    )
    source.done.clear()
    source.entered.clear()
    source.release.clear()
    started = time.monotonic()
    loader = remote.RemoteGreetings()
    try:
        assert time.monotonic() - started < 0.25
        assert source.entered.wait(1)
        assert loader.logos is None and not source.done.is_set()
    finally:
        source.release.set()
    assert source.done.wait(2)
    assert loader.logos == source.first.logos
    source.online = False
    assert _load(source).logos == source.first.logos
    source.online = True
    source.entered.clear()
    source.done.clear()
    source.release.clear()
    loader = remote.RemoteGreetings()
    try:
        assert source.entered.wait(1)
        assert loader.logos == source.first.logos
    finally:
        source.release.set()
    assert source.done.wait(2)
    assert loader.logos == second.logos
    assert any(path.read_bytes() == second.blob for path in source.cache.rglob(f"{second.reference.sha256}.json"))
    source.manifest["schema_version"] = 99
    for path in source.cache.rglob("manifest.json"):
        os.utime(path, (0, 0))
    assert _load(source).logos == second.logos
    source.online = False
    assert _load(source).logos == second.logos
    source.now = datetime(2026, 12, 28, tzinfo=timezone.utc)
    assert _load(source).logos is None


def test_encoding_cache_permissions_and_default_fallback(source):
    assert _load(source, encoding="ascii").logos is None
    source.cache.write_text("Cache directory is blocked by a file.", encoding="utf-8")
    assert _load(source).logos == source.first.logos
    source.manifest["collections"] = {"default": {"asset_ids": ["first"]}}
    del source.manifest["schedule"]
    source.requests.clear()
    assert _load(source).logos is None
    assert source.requests == ["manifest.json"]
