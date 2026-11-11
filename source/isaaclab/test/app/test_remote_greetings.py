# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scheduled artwork safety, optional caching, and nonblocking downloads."""

import hashlib
import io
import json
import os
import random
import threading
import time
import urllib.request
from datetime import date, datetime, timezone
from http.client import IncompleteRead
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import isaaclab.app._remote_greetings as remote
from isaaclab.app._greeting_catalog import AssetReference, decode_greeting, select_assets

pytestmark = pytest.mark.kitless


def _encode(payload):
    blob = json.dumps(payload, ensure_ascii=False).encode()
    digest = hashlib.sha256(blob).hexdigest()
    return blob, AssetReference("first", "holiday", f"assets/first.{digest[:16]}.json", digest, len(blob))


@pytest.fixture
def source(tmp_path, monkeypatch):
    logos = tuple(
        "\n".join(["\x1b[0;38;2;255;0;128;48;2;0;32;255m" + row + "\x1b[39;49m"] * 12)
        for row in ("新春" + " " * 20, "Season".ljust(74))
    )
    payload = {
        "schema_version": 1,
        "id": "first",
        "collection": "holiday",
        "slots": [{"cols": cols, "rows": 12, "frame": logo} for cols, logo in zip((24, 74), logos)],
    }
    blob, reference = _encode(payload)
    manifest = {
        "schema_version": 1,
        "active_collection": "default",
        "collections": {"default": {"asset_ids": []}, "holiday": {"asset_ids": ["first"]}},
        "assets": {"first": {"path": reference.path, "sha256": reference.sha256, "bytes": len(blob)}},
        "schedule": {
            "timezone": "UTC",
            "events": [
                {"id": "later", "collection": "holiday", "start": "2027-01-01", "end": "2027-01-02"},
                {"id": "winter", "collection": "holiday", "start": "2026-12-20", "end": "2026-12-27"},
            ],
        },
    }
    source = SimpleNamespace(
        cache=tmp_path / "cache",
        manifest=manifest,
        payload=payload,
        blob=blob,
        reference=reference,
        logos=logos,
        now=datetime(2026, 12, 24, tzinfo=timezone.utc),
        thread=threading.Thread,
        transport=remote._read_bytes,
    )
    source.read = Mock(
        side_effect=lambda url, limit: json.dumps(source.manifest).encode()
        if url.endswith("/manifest.json")
        else source.blob
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(source.cache))
    monkeypatch.setattr(remote, "datetime", SimpleNamespace(now=lambda tz: source.now.astimezone(tz)))
    monkeypatch.setattr(remote, "_read_bytes", source.read)
    monkeypatch.setattr(remote.threading, "Thread", lambda *, target, **kwargs: SimpleNamespace(start=target))
    return source


def test_utc_calendar_and_catalog_safety(source, monkeypatch):
    catalog = source.manifest
    for day, expected in ((19, ()), (20, ("first",)), (27, ("first",)), (28, ())):
        assert tuple(ref.identifier for ref in select_assets(catalog, date(2026, 12, day))) == expected
    for entry, key, value in (
        (catalog["assets"]["first"], "path", "assets/../first." + source.reference.sha256[:16] + ".json"),
        (catalog["schedule"], "timezone", "local"),
        (catalog["schedule"]["events"][1], "end", "2027-01-01"),
    ):
        with monkeypatch.context() as patch, pytest.raises(ValueError):
            patch.setitem(entry, key, value)
            select_assets(catalog, source.now.date())
    catalog["active_collection"] = "holiday"
    catalog["schedule"]["events"][1]["collection"] = "default"
    assert select_assets(catalog, source.now.date()) == ()
    del catalog["schedule"]
    assert select_assets(catalog, source.now.date()) == (source.reference,)
    catalog.update(active_collection="default", collections={"default": {"asset_ids": ["first"]}})
    assert remote.RemoteGreetings().logos is None
    assert source.read.call_count == 1


def test_static_artwork_integrity_and_terminal_safety(source, monkeypatch):
    assert decode_greeting(source.blob, source.reference) == source.logos
    for reference in (source.reference._replace(sha256="0" * 64), source.reference._replace(size=1)):
        with pytest.raises(ValueError):
            decode_greeting(source.blob, reference)
    slot = source.payload["slots"][0]
    for key, value in (
        ("frame", "\n".join(["新" * 24] * 12)),
        ("frames", [source.logos[0]]),
    ):
        with monkeypatch.context() as patch, pytest.raises(ValueError):
            patch.setitem(slot, key, value)
            decode_greeting(*_encode(source.payload))
    for control in ("\x1b[2J", "\x1b[1m", "\u2028"):
        with monkeypatch.context() as patch, pytest.raises(ValueError, match="terminal or Unicode controls|SGR"):
            patch.setitem(source.payload["slots"][1], "frame", control + source.logos[1])
            decode_greeting(*_encode(source.payload))


def test_optional_cache_fallbacks_preserve_random_state(source, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setenv("ISAACLAB_REMOTE_GREETINGS", "0")
        assert remote.RemoteGreetings().logos is None
    with monkeypatch.context() as patch:
        patch.delenv("XDG_CACHE_HOME")
        patch.delenv("LOCALAPPDATA", raising=False)
        patch.setattr(remote.Path, "home", Mock(side_effect=RuntimeError("No home directory")))
        assert remote.RemoteGreetings().logos is None
    source.read.assert_not_called()
    assert not source.cache.exists()
    assert remote.RemoteGreetings(encoding="ascii").logos is None
    state, valid = random.getstate(), source.blob
    source.blob = valid.replace(b"Season", b"Tamper")
    assert remote.RemoteGreetings().logos is None
    assert not source.cache.exists()
    source.blob = valid
    loader = remote.RemoteGreetings()
    assert loader.logos == source.logos
    manifest_path = next(source.cache.rglob("manifest.json"))
    source.manifest["assets"]["second"] = source.manifest["assets"]["first"]
    source.manifest["collections"]["holiday"]["asset_ids"].append("second")
    manifest_path.write_text(json.dumps(source.manifest))
    source.manifest["schema_version"] = 99
    loader._refresh()
    assert json.loads(manifest_path.read_bytes())["schema_version"] == 1
    for failure in (IncompleteRead(b"partial"), OSError("Offline")):
        source.read.side_effect = failure
        loader._refresh()
        os.utime(manifest_path, (0, 0))
        assert remote.RemoteGreetings().logos == source.logos
    source.now = datetime(2026, 12, 28, tzinfo=timezone.utc)
    assert remote.RemoteGreetings().logos is None
    assert random.getstate() == state


def test_hugging_face_download_does_not_block_startup(source, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    workers = []

    def worker(**kwargs):
        thread = source.thread(**kwargs)
        workers.append(thread)
        return thread

    def open_url(opener, request, **kwargs):
        root = "https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/greetings/"
        assert request.full_url in (root + "manifest.json", root + source.reference.path)
        entered.set()
        if not release.wait(2):
            raise TimeoutError("Test did not release its response")
        return io.BytesIO(source.read(request.full_url, 0))

    monkeypatch.setattr(remote.threading, "Thread", worker)
    monkeypatch.setattr(remote, "_read_bytes", source.transport)
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", open_url)
    started = time.monotonic()
    try:
        loader = remote.RemoteGreetings()
        assert time.monotonic() - started < 0.25
        assert entered.wait(1) and loader.logos is None
    finally:
        release.set()
        for thread in workers:
            thread.join(2)
    assert not workers[0].is_alive()
    assert loader.logos == source.logos
