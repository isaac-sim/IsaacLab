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
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

import pytest

import isaaclab.app._remote_greetings as remote_greetings

pytestmark = pytest.mark.kitless

_TODAY = datetime.date(2026, 12, 24)


class _Source(NamedTuple):
    root: Path
    cache: Path
    logos: tuple[str, str]
    asset_path: str


def _set_date(monkeypatch, on_date: datetime.date) -> None:
    class FrozenDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.datetime.combine(on_date, datetime.time(), tzinfo=datetime.timezone.utc)

    monkeypatch.setattr(remote_greetings, "datetime", FrozenDateTime)


@pytest.fixture
def greeting_source(tmp_path, monkeypatch):
    root = tmp_path / "remote"
    cache = tmp_path / "cache"
    directory = root / "Greetings"
    (directory / "assets").mkdir(parents=True)
    logos = (
        "\n".join(["\x1b[38;2;255;0;128m新春" + " " * 20 + "\x1b[0m"] * 12),
        "\n".join(["\x1b[38;2;0;255;0mSeason" + " " * 68 + "\x1b[0m"] * 12),
    )
    payload = json.dumps(
        {
            "schema_version": 1,
            "id": "seasonal",
            "collection": "holiday",
            "slots": [
                {"cols": 24, "rows": 12, "frame_duration_ms": 100, "frames": [logos[0]]},
                {"cols": 74, "rows": 12, "frame_duration_ms": 100, "frames": [logos[1]]},
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    asset_path = f"assets/seasonal.{digest[:16]}.json"
    (directory / asset_path).write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "active_collection": "default",
        "collections": {"default": {"asset_ids": []}, "holiday": {"asset_ids": ["seasonal"]}},
        "assets": {"seasonal": {"path": asset_path, "sha256": digest, "bytes": len(payload)}},
        "schedule": {
            "timezone": "UTC",
            "events": [{"id": "winter", "collection": "holiday", "start": "2026-12-24", "end": "2026-12-24"}],
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    monkeypatch.setattr(remote_greetings, "ISAACLAB_NUCLEUS_DIR", str(root))
    _set_date(monkeypatch, _TODAY)
    return _Source(root, cache, logos, asset_path)


def _wait_for_logos(loader, expected: tuple[str, str]) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if loader.logos == expected:
            return
        threading.Event().wait(0.005)
    assert loader.logos == expected


def _assert_logos_stay(loader, expected) -> None:
    deadline = time.monotonic() + 0.1
    while time.monotonic() < deadline:
        assert loader.logos == expected
        threading.Event().wait(0.005)


def _make_cache_stale(cache: Path) -> None:
    for path in cache.rglob("*"):
        if path.is_file():
            os.utime(path, (0, 0))


class _ObservedResponse(io.BytesIO):
    def __init__(self, blob: bytes, finished: threading.Event):
        super().__init__(blob)
        self._finished = finished

    def close(self) -> None:
        super().close()
        self._finished.set()


def _serve_http(monkeypatch, source: _Source, release: threading.Event | None = None):
    requested = []
    entered = threading.Event()
    manifest_read = threading.Event()
    asset_read = threading.Event()

    def open_url(opener, request, *args, **kwargs):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        relative = urlsplit(url).path.removeprefix("/IsaacLab/Greetings/")
        requested.append(relative)
        entered.set()
        if release is not None and not release.wait(3.0):
            raise TimeoutError("The test did not release its blocked HTTP response.")
        blob = (source.root / "Greetings" / relative).read_bytes()
        return _ObservedResponse(blob, asset_read if relative.startswith("assets/") else manifest_read)

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", open_url)
    monkeypatch.setattr(remote_greetings, "ISAACLAB_NUCLEUS_DIR", "https://greetings.example/IsaacLab")
    return requested, entered, manifest_read, asset_read


def test_cold_start_returns_before_http_and_publishes_during_startup(greeting_source, monkeypatch):
    release = threading.Event()
    _, entered, _, _ = _serve_http(monkeypatch, greeting_source, release)
    returned = threading.Event()
    result = {}

    def construct():
        try:
            result["loader"] = remote_greetings.RemoteGreetings()
        finally:
            returned.set()

    threading.Thread(target=construct, daemon=True).start()
    try:
        assert entered.wait(1.0), "The refresh never reached the HTTP transport."
        assert returned.wait(0.25), "Greeting construction waited for the HTTP response."
        assert result["loader"].logos is None
    finally:
        release.set()
    assert returned.wait(2.0)
    _wait_for_logos(result["loader"], greeting_source.logos)


def test_cached_greeting_is_available_offline_and_expires_by_utc_date(greeting_source, monkeypatch):
    _wait_for_logos(remote_greetings.RemoteGreetings(), greeting_source.logos)
    shutil.rmtree(greeting_source.root)

    assert remote_greetings.RemoteGreetings().logos == greeting_source.logos

    _set_date(monkeypatch, datetime.date(2026, 12, 25))
    assert remote_greetings.RemoteGreetings().logos is None


def test_offline_startup_uses_a_cached_design_when_other_current_designs_are_missing(greeting_source, monkeypatch):
    directory = greeting_source.root / "Greetings"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = json.loads((directory / greeting_source.asset_path).read_bytes())
    for identifier in ("uncached-a", "uncached-b"):
        payload = {**original, "id": identifier}
        blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(blob).hexdigest()
        asset_path = f"assets/{identifier}.{digest[:16]}.json"
        (directory / asset_path).write_bytes(blob)
        manifest["assets"][identifier] = {"path": asset_path, "sha256": digest, "bytes": len(blob)}
    manifest["collections"]["holiday"]["asset_ids"] = ["uncached-a", "uncached-b", "seasonal"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Download only the seasonal design, then try an uncached design first offline.
    monkeypatch.setattr(remote_greetings, "random", random.Random(5))
    _wait_for_logos(remote_greetings.RemoteGreetings(), greeting_source.logos)
    shutil.rmtree(greeting_source.root)
    monkeypatch.setattr(remote_greetings, "random", random.Random(0))

    assert remote_greetings.RemoteGreetings().logos == greeting_source.logos


def test_cached_fallback_does_not_prevent_downloading_the_next_chosen_design(greeting_source, monkeypatch):
    directory = greeting_source.root / "Greetings"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    alternate_logos = (
        greeting_source.logos[0].replace("新春", "元旦"),
        greeting_source.logos[1].replace("Season", "Spring"),
    )
    payload = json.loads((directory / greeting_source.asset_path).read_bytes())
    payload["id"] = "alternate"
    for slot, logo in zip(payload["slots"], alternate_logos):
        slot["frames"] = [logo]
    blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    asset_path = f"assets/alternate.{digest[:16]}.json"
    (directory / asset_path).write_bytes(blob)
    manifest["assets"]["alternate"] = {"path": asset_path, "sha256": digest, "bytes": len(blob)}
    manifest["collections"]["holiday"]["asset_ids"] = ["seasonal", "alternate"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _serve_http(monkeypatch, greeting_source)
    monkeypatch.setattr(remote_greetings, "random", random.Random(1))
    _wait_for_logos(remote_greetings.RemoteGreetings(), greeting_source.logos)
    release = threading.Event()
    _, entered, _, _ = _serve_http(monkeypatch, greeting_source, release)
    # The next startup requests alternate first while only seasonal is cached.
    monkeypatch.setattr(remote_greetings, "random", random.Random(1))

    loader = remote_greetings.RemoteGreetings()
    try:
        assert loader.logos == greeting_source.logos
        assert entered.wait(1.0), "A cached fallback prevented refreshing the uncached chosen design."
        assert loader.logos == greeting_source.logos
    finally:
        release.set()
    _wait_for_logos(loader, alternate_logos)
    assert any(path.read_bytes() == blob for path in greeting_source.cache.rglob(f"{digest}.json"))


def test_invalid_refresh_preserves_the_last_valid_cached_manifest(greeting_source, monkeypatch):
    _, _, manifest_read, _ = _serve_http(monkeypatch, greeting_source)
    _wait_for_logos(remote_greetings.RemoteGreetings(), greeting_source.logos)
    (greeting_source.root / "Greetings" / "manifest.json").write_text('{"schema_version": 99}', encoding="utf-8")
    _make_cache_stale(greeting_source.cache)
    manifest_read.clear()

    loader = remote_greetings.RemoteGreetings()
    assert loader.logos == greeting_source.logos
    assert manifest_read.wait(1.0), "The stale catalog was not refreshed."
    _assert_logos_stay(loader, greeting_source.logos)
    shutil.rmtree(greeting_source.root)

    assert remote_greetings.RemoteGreetings().logos == greeting_source.logos


def test_bad_payload_is_not_displayed_or_trusted_in_cache(greeting_source, monkeypatch):
    _, _, _, asset_read = _serve_http(monkeypatch, greeting_source)
    asset = greeting_source.root / "Greetings" / greeting_source.asset_path
    valid_payload = asset.read_bytes()
    invalid_payload = b"x" * len(valid_payload)
    asset.write_bytes(invalid_payload)

    loader = remote_greetings.RemoteGreetings()
    assert asset_read.wait(1.0), "The selected greeting payload was not requested."
    _assert_logos_stay(loader, None)
    assert all(path.read_bytes() != invalid_payload for path in greeting_source.cache.rglob("*") if path.is_file())

    asset.write_bytes(valid_payload)
    _make_cache_stale(greeting_source.cache)
    _wait_for_logos(remote_greetings.RemoteGreetings(), greeting_source.logos)


def test_default_collection_never_fetches_remote_art(greeting_source, monkeypatch):
    manifest_path = greeting_source.root / "Greetings" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["collections"] = {"default": {"asset_ids": ["seasonal"]}}
    del manifest["schedule"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    requested, _, manifest_read, _ = _serve_http(monkeypatch, greeting_source)

    loader = remote_greetings.RemoteGreetings()
    assert manifest_read.wait(1.0), "The catalog was not requested."
    _assert_logos_stay(loader, None)
    assert not any(path.startswith("assets/") for path in requested)


def test_unwritable_cache_does_not_prevent_displaying_downloaded_art(greeting_source):
    greeting_source.cache.write_text("A file blocks creation of the cache directory.", encoding="utf-8")

    _wait_for_logos(remote_greetings.RemoteGreetings(), greeting_source.logos)


def test_terminal_encoding_must_support_both_greeting_slots(greeting_source, monkeypatch):
    _, _, _, asset_read = _serve_http(monkeypatch, greeting_source)

    loader = remote_greetings.RemoteGreetings(encoding="ascii")
    assert asset_read.wait(1.0), "The selected greeting payload was not requested."
    _assert_logos_stay(loader, None)
