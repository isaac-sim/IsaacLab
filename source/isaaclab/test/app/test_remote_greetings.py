# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for scheduled greetings, validated artwork, and offline startup."""

import hashlib
import io
import json
import os
import random
import shutil
import threading
import urllib.request
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, timezone
from types import SimpleNamespace
from urllib.parse import urlsplit

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
    (source.directory / reference.path).write_bytes(blob)
    source.manifest["assets"][identifier] = {"path": reference.path, "sha256": reference.sha256, "bytes": len(blob)}
    source.manifest["collections"]["holiday"]["asset_ids"].append(identifier)
    source.catalog.write_text(json.dumps(source.manifest), encoding="utf-8")
    return SimpleNamespace(logos=logos, payload=payload, blob=blob, reference=reference)


@pytest.fixture
def source(tmp_path, monkeypatch):
    source = SimpleNamespace(
        directory=tmp_path / "remote" / "Greetings",
        cache=tmp_path / "cache",
        now=datetime(2026, 12, 24, tzinfo=timezone.utc),
        requests=[],
        entered=threading.Event(),
        release=threading.Event(),
        done=threading.Event(),
    )
    (source.directory / "assets").mkdir(parents=True)
    source.catalog = source.directory / "manifest.json"
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

    @contextmanager
    def open_url(opener, request, *args, **kwargs):
        path = urlsplit(request.full_url).path.removeprefix("/IsaacLab/Greetings/")
        source.requests.append(path)
        source.entered.set()
        if not source.release.wait(2):
            raise TimeoutError("Test did not release its HTTP response.")
        with io.BytesIO((source.directory / path).read_bytes()) as response:
            yield response

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


def test_utc_calendar_boundaries_order_and_fallback(source):
    _add_asset(source, "second")
    source.manifest["assets"] = dict(reversed(source.manifest["assets"].items()))
    for day, identifiers in ((19, ()), (20, ("first", "second")), (27, ("first", "second")), (28, ())):
        assert tuple(ref.identifier for ref in select_assets(source.manifest, date(2026, 12, day))) == identifiers, day
    source.manifest["active_collection"] = "holiday"
    source.manifest["schedule"]["events"][1]["collection"] = "default"
    assert select_assets(source.manifest, source.now.date()) == ()
    assert len(select_assets(source.manifest, date(2026, 1, 1))) == 2
    del source.manifest["schedule"]
    assert len(select_assets(source.manifest, source.now.date())) == 2


def test_invalid_catalogs(source):
    for field, value in (
        ("path", "assets/../first." + source.first.reference.sha256[:16] + ".json"),
        ("path", "https://example.com/" + source.first.reference.path),
        ("sha256", "0" * 64),
        ("bytes", 262145),
    ):
        manifest = deepcopy(source.manifest)
        manifest["assets"]["first"][field] = value
        with pytest.raises(ValueError):
            select_assets(manifest, date(2026, 12, 19))
    for field, value in (
        ("start", "20261220"),
        ("end", "2026-12-19"),
        ("id", "later"),
        ("collection", "missing"),
        ("end", "2027-01-01"),
    ):
        manifest = deepcopy(source.manifest)
        manifest["schedule"]["events"][1][field] = value
        with pytest.raises(ValueError):
            select_assets(manifest, source.now.date())
    for change in (
        {"schema_version": 2},
        {"schedule": {"timezone": "local", "events": []}},
        {"collections": {"default": {"asset_ids": []}, "holiday": {"asset_ids": ["first", "first"]}}},
        {"collections": {"default": {"asset_ids": []}, "holiday": {"asset_ids": ["missing"]}}},
    ):
        with pytest.raises(ValueError):
            select_assets({**source.manifest, **change}, source.now.date())


def test_payload_integrity_and_static_slots(source):
    blob, reference = _encode(source.first.payload)
    assert decode_greeting(blob, reference) == source.first.logos
    for invalid in (reference._replace(size=len(blob) + 1), reference._replace(sha256="0" * 64)):
        with pytest.raises(ValueError):
            decode_greeting(blob, invalid)
    for change in (
        {"schema_version": 2},
        {"id": "wrong"},
        {"collection": "default"},
        {"padding": " " * 262144},
        {"slots": source.first.payload["slots"][::-1]},
    ):
        with pytest.raises(ValueError):
            decode_greeting(*_encode({**source.first.payload, **change}))
    for change in (
        {"frame": "\n".join(["新" * 24] * 12)},
        {"frame": [source.first.logos[0]]},
        {"frame_duration_ms": 100, "frames": [source.first.logos[0]]},
    ):
        payload = deepcopy(source.first.payload)
        payload["slots"][0].update(change)
        with pytest.raises(ValueError):
            decode_greeting(*_encode(payload))
    with pytest.raises(ValueError):
        decode_greeting(*_encode([]))


@pytest.mark.parametrize(
    "control", ["\x1b[2J", "\x1b]8;;x\x07", "\x1b[1m", "\x1b[38;2;256;0;0m", "\r", "\x9b", "\u2028", "\u2029"]
)
def test_terminal_controls(source, control):
    source.first.payload["slots"][1]["frame"] = "\n".join([" " * 37 + control + " " * 37] * 12)
    with pytest.raises(ValueError):
        decode_greeting(*_encode(source.first.payload))


def test_startup_cache_variety_and_utc_expiry(source, monkeypatch):
    second = _add_asset(source, "second", "Spring")
    _add_asset(source, "third", "Winter")
    monkeypatch.setattr(remote, "random", random.Random(1))
    source.release.clear()
    result = []
    startup = threading.Thread(target=lambda: result.append(remote.RemoteGreetings()), daemon=True)
    startup.start()
    try:
        assert source.entered.wait(1)
        startup.join(0.25)
        assert result and result[0].logos is None
    finally:
        source.release.set()
    assert source.done.wait(2)
    assert result[0].logos == source.first.logos
    offline = source.directory.with_name("offline")
    source.directory.rename(offline)
    monkeypatch.setattr(remote, "random", random.Random(6))
    assert _load(source).logos == source.first.logos
    offline.rename(source.directory)
    monkeypatch.setattr(remote, "random", random.Random(6))
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
    source.now = datetime(2026, 12, 28, tzinfo=timezone.utc)
    assert remote.RemoteGreetings().logos is None


def test_corrupt_refresh_preserves_valid_offline_cache(source):
    corrupt = source.first.blob.replace(b"Season", b"Tamper")
    asset = source.directory / source.first.reference.path
    asset.write_bytes(corrupt)
    assert _load(source).logos is None
    assert all(path.read_bytes() != corrupt for path in source.cache.rglob("*") if path.is_file())
    asset.write_bytes(source.first.blob)
    assert _load(source).logos == source.first.logos
    source.catalog.write_text('{"schema_version": 99}', encoding="utf-8")
    for path in source.cache.rglob("manifest.json"):
        os.utime(path, (0, 0))
    assert _load(source).logos == source.first.logos
    shutil.rmtree(source.directory)
    assert _load(source).logos == source.first.logos


def test_encoding_cache_permissions_and_default_fallback(source):
    assert _load(source, encoding="ascii").logos is None
    source.cache.write_text("Cache directory is blocked by a file.", encoding="utf-8")
    assert _load(source).logos == source.first.logos
    source.manifest["collections"] = {"default": {"asset_ids": ["first"]}}
    del source.manifest["schedule"]
    source.catalog.write_text(json.dumps(source.manifest), encoding="utf-8")
    source.requests.clear()
    assert _load(source).logos is None
    assert source.requests == ["manifest.json"]
