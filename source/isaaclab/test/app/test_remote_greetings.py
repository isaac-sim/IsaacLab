# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Startup and persistence contracts for remote greeting I/O."""

import datetime
import hashlib
import io
import json
import os
import random
import shutil
import threading
import time
import urllib.request
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

import isaaclab.app._remote_greetings as remote_greetings

pytestmark = pytest.mark.kitless


def _set_date(monkeypatch, day):
    class FrozenDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime(2026, 12, day, tzinfo=datetime.timezone.utc)

    monkeypatch.setattr(remote_greetings, "datetime", FrozenDateTime)


def _add_asset(source, identifier, label="Season"):
    logos = tuple(
        "\n".join([f"\x1b[38;2;255;0;128m{text}{' ' * padding}\x1b[0m"] * 12)
        for text, padding in (("新春", 20), (label, 74 - len(label)))
    )
    payload = {
        "schema_version": 1,
        "id": identifier,
        "collection": "holiday",
        "slots": [
            {"cols": cols, "rows": 12, "frame_duration_ms": 100, "frames": [logo]}
            for cols, logo in zip((24, 74), logos)
        ],
    }
    blob = json.dumps(payload, ensure_ascii=False).encode()
    digest = hashlib.sha256(blob).hexdigest()
    path = f"assets/{identifier}.{digest[:16]}.json"
    (source.directory / path).write_bytes(blob)
    source.manifest["assets"][identifier] = {"path": path, "sha256": digest, "bytes": len(blob)}
    source.manifest["collections"]["holiday"]["asset_ids"].append(identifier)
    source.catalog.write_text(json.dumps(source.manifest), encoding="utf-8")
    return logos, path, blob


@pytest.fixture
def source(tmp_path, monkeypatch):
    source = SimpleNamespace(directory=tmp_path / "remote" / "Greetings", cache=tmp_path / "cache")
    (source.directory / "assets").mkdir(parents=True)
    source.catalog = source.directory / "manifest.json"
    source.manifest = {
        "schema_version": 1,
        "active_collection": "default",
        "collections": {"default": {"asset_ids": []}, "holiday": {"asset_ids": []}},
        "assets": {},
        "schedule": {
            "timezone": "UTC",
            "events": [{"id": "winter", "collection": "holiday", "start": "2026-12-24", "end": "2026-12-24"}],
        },
    }
    source.logos, source.asset, source.payload = _add_asset(source, "seasonal")
    monkeypatch.setenv("XDG_CACHE_HOME", str(source.cache))
    monkeypatch.setattr(remote_greetings, "ISAACLAB_NUCLEUS_DIR", str(source.directory.parent))
    _set_date(monkeypatch, 24)
    return source


def _wait_for_logos(loader, expected):
    deadline = time.monotonic() + 2
    while loader.logos != expected and time.monotonic() < deadline:
        time.sleep(0.005)
    assert loader.logos == expected


def _assert_logos_stay(loader, expected):
    deadline = time.monotonic() + 0.1
    while time.monotonic() < deadline:
        assert loader.logos == expected
        time.sleep(0.005)


def _serve_http(monkeypatch, source, release=None):
    requested, entered, finished = [], threading.Event(), threading.Event()

    @contextmanager
    def open_url(opener, request, *args, **kwargs):
        relative = urlsplit(request.full_url).path.removeprefix("/IsaacLab/Greetings/")
        requested.append(relative)
        entered.set()
        if release is not None and not release.wait(3):
            raise TimeoutError("The test did not release its HTTP response.")
        with io.BytesIO((source.directory / relative).read_bytes()) as response:
            yield response
        finished.set()

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", open_url)
    monkeypatch.setattr(remote_greetings, "ISAACLAB_NUCLEUS_DIR", "https://greetings.example/IsaacLab")
    return requested, entered, finished


def test_cold_start_returns_before_http_and_publishes_during_startup(source, monkeypatch):
    release, result = threading.Event(), []
    _, entered, _ = _serve_http(monkeypatch, source, release)
    startup = threading.Thread(target=lambda: result.append(remote_greetings.RemoteGreetings()), daemon=True)
    startup.start()
    try:
        assert entered.wait(1)
        startup.join(0.25)
        assert result, "Greeting construction waited for the HTTP response."
        assert result[0].logos is None
    finally:
        release.set()
    _wait_for_logos(result[0], source.logos)


def test_partial_cache_is_available_offline_and_expires_by_utc_date(source, monkeypatch):
    for identifier in ("alternate", "third"):
        _add_asset(source, identifier)
    monkeypatch.setattr(remote_greetings, "random", random.Random(1))
    _wait_for_logos(remote_greetings.RemoteGreetings(), source.logos)
    shutil.rmtree(source.directory)
    # Seed 6 tries an uncached design first with either choice or shuffle.
    monkeypatch.setattr(remote_greetings, "random", random.Random(6))
    assert remote_greetings.RemoteGreetings().logos == source.logos
    _set_date(monkeypatch, 25)
    assert remote_greetings.RemoteGreetings().logos is None


def test_cached_fallback_still_downloads_the_next_chosen_design(source, monkeypatch):
    alternate, _, blob = _add_asset(source, "alternate", "Spring")
    _serve_http(monkeypatch, source)
    monkeypatch.setattr(remote_greetings, "random", random.Random(1))
    _wait_for_logos(remote_greetings.RemoteGreetings(), source.logos)
    release = threading.Event()
    _, entered, _ = _serve_http(monkeypatch, source, release)
    monkeypatch.setattr(remote_greetings, "random", random.Random(1))
    loader = remote_greetings.RemoteGreetings()
    try:
        assert entered.wait(1), "Cached fallback prevented fetching the uncached chosen design."
        assert loader.logos == source.logos
    finally:
        release.set()
    _wait_for_logos(loader, alternate)
    assert any(cached.read_bytes() == blob for cached in source.cache.rglob(f"{hashlib.sha256(blob).hexdigest()}.json"))


def test_invalid_refresh_preserves_valid_cache(source, monkeypatch):
    _, _, finished = _serve_http(monkeypatch, source)
    _wait_for_logos(remote_greetings.RemoteGreetings(), source.logos)
    source.catalog.write_text('{"schema_version": 99}', encoding="utf-8")
    for path in source.cache.rglob("manifest.json"):
        os.utime(path, (0, 0))
    finished.clear()
    loader = remote_greetings.RemoteGreetings()
    assert finished.wait(1), "The stale catalog was not refreshed."
    _assert_logos_stay(loader, source.logos)
    shutil.rmtree(source.directory)
    assert remote_greetings.RemoteGreetings().logos == source.logos


def test_corrupt_payload_is_not_displayed_or_cached(source, monkeypatch):
    requested, _, _ = _serve_http(monkeypatch, source)
    corrupt = source.payload.replace(b"Season", b"Tamper")
    (source.directory / source.asset).write_bytes(corrupt)
    loader = remote_greetings.RemoteGreetings()
    deadline = time.monotonic() + 1
    while source.asset not in requested and time.monotonic() < deadline:
        time.sleep(0.005)
    assert source.asset in requested
    _assert_logos_stay(loader, None)
    assert all(path.read_bytes() != corrupt for path in source.cache.rglob("*") if path.is_file())
    (source.directory / source.asset).write_bytes(source.payload)
    _wait_for_logos(remote_greetings.RemoteGreetings(), source.logos)


def test_default_collection_never_fetches_remote_art(source, monkeypatch):
    source.manifest["collections"] = {"default": {"asset_ids": ["seasonal"]}}
    del source.manifest["schedule"]
    source.catalog.write_text(json.dumps(source.manifest), encoding="utf-8")
    requested, _, finished = _serve_http(monkeypatch, source)
    loader = remote_greetings.RemoteGreetings()
    assert finished.wait(1)
    _assert_logos_stay(loader, None)
    assert not any(path.startswith("assets/") for path in requested)


def test_encoding_and_unwritable_cache_do_not_break_startup(source, monkeypatch):
    requested, _, finished = _serve_http(monkeypatch, source)
    loader = remote_greetings.RemoteGreetings(encoding="ascii")
    assert finished.wait(1)
    _assert_logos_stay(loader, None)
    assert source.asset in requested
    source.cache.write_text("A file blocks cache directory creation.", encoding="utf-8")
    _wait_for_logos(remote_greetings.RemoteGreetings(), source.logos)
