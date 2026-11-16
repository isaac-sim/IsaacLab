# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cache seasonal loading-screen artwork and refresh it without blocking startup."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import tempfile
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.client import HTTPException, HTTPMessage, HTTPResponse
from pathlib import Path
from urllib.parse import urlsplit

from ._greeting_catalog import AssetReference, decode_greeting, select_assets

logger = logging.getLogger(__name__)

_MAX_CATALOG_BYTES = 524288
_MAX_ASSET_BYTES = 262144
_REQUEST_TIMEOUT = 1.0
_REFRESH_INTERVAL = 300
_GREETINGS_ROOT = "https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/greetings"


class RemoteGreetings:
    """Read a cached greeting pair immediately and refresh it in a daemon thread.

    The default collection deliberately yields no remote artwork, leaving the
    loading screen's bundled greeting in place. Cache and network failures are
    optional: they never prevent a screen from opening or a process from exiting.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self._lock = threading.Lock()
        self._logos: tuple[str, str] | None = None
        if os.getenv("ISAACLAB_REMOTE_GREETINGS") == "0":
            return
        self._root = _GREETINGS_ROOT
        try:
            cache_root = os.getenv("XDG_CACHE_HOME") or os.getenv("LOCALAPPDATA") or str(Path.home() / ".cache")
        except (OSError, RuntimeError) as error:
            logger.debug("Loading greeting cache directory unavailable: %s", error)
            return
        namespace = hashlib.sha256(self._root.encode()).hexdigest()[:16]
        self._cache = Path(cache_root) / "isaaclab" / "greetings" / namespace
        self._manifest_path = self._cache / "manifest.json"
        self._on_date = datetime.now(timezone.utc).date()
        self._encoding = encoding
        self._reference: AssetReference | None = None
        self._random = random.Random()
        fresh = False
        try:
            manifest = json.loads(_read_local(self._manifest_path, _MAX_CATALOG_BYTES).decode("utf-8"))
            references = list(select_assets(manifest, self._on_date))
            self._random.shuffle(references)
            desired_cached = not references
            if references:
                self._reference = references[0]
            for reference in references:
                try:
                    logos = self._cached_pair(reference)
                except (OSError, ValueError, RecursionError):
                    continue
                self._logos = logos
                desired_cached = reference == self._reference
                break
            age = time.time() - self._manifest_path.stat().st_mtime
            fresh = 0 <= age < _REFRESH_INTERVAL and desired_cached
        except (OSError, ValueError, RecursionError) as error:
            logger.debug("Cached loading greeting unavailable: %s", error)
        if not fresh:
            threading.Thread(target=self._refresh, name="isaaclab-greetings", daemon=True).start()

    @property
    def logos(self) -> tuple[str, str] | None:
        """Return one validated, encodable narrow/wide pair, or no seasonal override."""
        with self._lock:
            return self._logos

    def _cached_pair(self, reference: AssetReference) -> tuple[str, str]:
        """Validate cached bytes against the selected catalog entry."""
        blob = _read_local(self._cache / f"{reference.sha256}.json", _MAX_ASSET_BYTES)
        return self._encodable_pair(blob, reference)

    def _encodable_pair(self, blob: bytes, reference: AssetReference) -> tuple[str, str]:
        """Reject artwork the active console cannot represent without substitutions."""
        logos = decode_greeting(blob, reference)
        for logo in logos:
            logo.encode(self._encoding)
        return logos

    def _refresh(self) -> None:
        """Publish a new pair only after validating its catalog and selected payload."""
        try:
            manifest_blob = _read_bytes(f"{self._root}/manifest.json", _MAX_CATALOG_BYTES)
            manifest = json.loads(manifest_blob.decode("utf-8"))
            references = select_assets(manifest, self._on_date)
            logos = None
            if references:
                # Keep the same design on a routine refresh while allowing new revisions.
                reference = next(
                    (item for item in references if self._reference and item.identifier == self._reference.identifier),
                    None,
                ) or self._random.choice(references)
                try:
                    logos = self._cached_pair(reference)
                except (OSError, ValueError):
                    blob = _read_bytes(f"{self._root}/{reference.path}", _MAX_ASSET_BYTES)
                    logos = self._encodable_pair(blob, reference)
                    self._write_cache(self._cache / f"{reference.sha256}.json", blob)
            # A bad payload must not replace a catalog that can still render cached art.
            self._write_cache(self._manifest_path, manifest_blob)
            with self._lock:
                self._logos = logos
        except (OSError, ValueError, RecursionError, HTTPException) as error:
            logger.debug("Remote loading greeting unavailable: %s", error)

    def _write_cache(self, path: Path, blob: bytes) -> None:
        """Atomically persist valid data when the user has a writable cache."""
        temporary = None
        try:
            self._cache.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self._cache, delete=False) as output:
                temporary = Path(output.name)
                output.write(blob)
            os.replace(temporary, path)
        except OSError as error:
            logger.debug("Loading greeting cache is not writable: %s", error)
        finally:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    temporary.unlink()


def _read_local(path: Path, limit: int) -> bytes:
    """Bound disk reads as well as network payloads, including previously cached data."""
    with path.open("rb") as stream:
        blob = stream.read(limit + 1)
    if len(blob) > limit:
        raise ValueError("Loading greeting data exceeds its byte limit.")
    return blob


def _read_bytes(location: str, limit: int) -> bytes:
    """Read a bounded HTTP response without importing a simulator or an image library."""
    request = urllib.request.Request(location, headers={"Accept": "application/json"})
    opener = urllib.request.build_opener(_SameOriginRedirect())
    with opener.open(request, timeout=_REQUEST_TIMEOUT) as response:
        deadline = time.monotonic() + _REQUEST_TIMEOUT
        chunks = []
        size = 0
        while size <= limit:
            if time.monotonic() >= deadline:
                raise TimeoutError("Loading greeting download exceeded its time limit.")
            chunk = response.read1(min(65536, limit + 1 - size))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            size += len(chunk)
    raise ValueError("Loading greeting download exceeds its byte limit.")


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    """Keep optional banner downloads on their configured asset origin."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: HTTPResponse,
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        if urlsplit(req.full_url)[:2] != urlsplit(newurl)[:2]:
            raise ValueError("A loading greeting redirected outside its asset origin.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)
