# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral checks for remote documentation images using a local HTTP server."""

from __future__ import annotations

import contextlib
import http.server
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tools.docs.check_media import check_image, collect_images


@pytest.fixture
def image_server():
    """Serve controlled HTTP responses, retaining the final response for subsequent requests."""
    routes = {}
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            responses = routes[self.path]
            status, headers, body, delay = responses.pop(0) if len(responses) > 1 else responses[0]
            time.sleep(delay)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            # The timeout case deliberately closes the client before the response.
            with contextlib.suppress(BrokenPipeError):
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", routes, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_collect_remote_images_and_source_locations(tmp_path):
    """Both directives retain query strings; duplicate URLs and local assets need no extra fetch."""
    path = tmp_path / "guide.rst"
    path.write_text(
        ".. figure:: https://cdn.example/robot.gif?v=2\n"
        "   :alt: Robot\n"
        "  .. image:: https://cdn.example/robot.gif?v=2\n"
        ".. image:: _static/local.png\n"
        ".. image:: https://cdn.example/chart.png\n"
        "https://example.org/ordinary-link\n"
    )
    assert collect_images([path]) == {
        "https://cdn.example/robot.gif?v=2": (path, 1),
        "https://cdn.example/chart.png": (path, 5),
    }


def test_asset_becomes_available_after_sync(image_server):
    """A 404 followed by a GIF succeeds inside the bounded recheck window."""
    origin, routes, requests = image_server
    routes["/robot.gif"] = [
        (404, {"Content-Type": "text/html"}, b"Not found", 0),
        (200, {"Content-Type": "image/gif"}, b"GIF89a", 0),
    ]
    assert check_image(origin + "/robot.gif", attempts=2, retry_delay=0) is None
    assert requests == ["/robot.gif", "/robot.gif"]


@pytest.mark.parametrize(
    "status,content_type,body,reason",
    [
        (404, "text/html", b"Not found", "HTTP 404"),
        (200, "text/html", b"<html>Error</html>", "not a supported image"),
        (200, "image/gif", b"<html>Error</html>", "not a supported image"),
        (200, "image/svg+xml", b"<html><svg></svg>Error</html>", "not a supported image"),
    ],
)
def test_unavailable_or_html_response_is_rejected(image_server, status, content_type, body, reason):
    """Persistent missing assets and HTML fallback pages cannot be mistaken for loaded images."""
    origin, routes, requests = image_server
    routes["/image"] = [(status, {"Content-Type": content_type}, body, 0)]
    assert reason in check_image(origin + "/image", attempts=2, retry_delay=0)
    assert len(requests) == 2


def test_redirect_to_image(image_server):
    """A hosted URL redirecting to a valid image succeeds."""
    origin, routes, _ = image_server
    routes["/redirect"] = [(302, {"Location": origin + "/image"}, b"", 0)]
    routes["/image"] = [(200, {"Content-Type": "image/png"}, b"\x89PNG\r\n\x1a\n", 0)]
    assert check_image(origin + "/redirect", attempts=1) is None


@pytest.mark.parametrize(
    "content_type,body",
    [
        ("image/gif", b"GIF87a"),
        ("image/jpeg", b"\xff\xd8\xff"),
        ("image/webp", b"RIFF\x00\x00\x00\x00WEBP"),
        ("image/svg+xml", b'<?xml version="1.0"?><!-- image --><svg xmlns="http://www.w3.org/2000/svg"/>'),
    ],
)
def test_supported_image_signatures(image_server, content_type, body):
    """Format-specific validation accepts the other documented image signatures."""
    origin, routes, _ = image_server
    routes["/image"] = [(200, {"Content-Type": content_type}, body, 0)]
    assert check_image(origin + "/image", attempts=1) is None


def test_timeout_is_bounded(image_server):
    """An unresponsive host reports a failure within the configured request timeout."""
    origin, routes, _ = image_server
    routes["/slow"] = [(200, {"Content-Type": "image/gif"}, b"GIF89a", 0.1)]
    assert "request failed" in check_image(origin + "/slow", attempts=1, timeout=0.01)


def test_cli_warning_and_strict_followup(image_server, tmp_path):
    """The same unavailable asset warns on a PR and fails a strict scheduled/manual follow-up."""
    origin, routes, _ = image_server
    routes["/missing.gif"] = [(404, {"Content-Type": "text/html"}, b"Not found", 0)]
    path = tmp_path / "guide.rst"
    path.write_text(f".. figure:: {origin}/missing.gif\n")
    command = [sys.executable, str(Path(__file__).with_name("check_media.py")), str(path), "--attempts", "1"]
    warning = subprocess.run([*command, "--warn-only"], capture_output=True, text=True, timeout=10)
    strict = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert warning.returncode == 0 and "::warning file=" in warning.stdout
    assert strict.returncode == 1 and "::error file=" in strict.stdout
    assert "line=1" in warning.stdout and "HTTP 404" in warning.stdout


def test_cli_limits_pr_checks_to_changed_rst_files(image_server, tmp_path):
    """A PR checks its new image without fetching unchanged images or opening deleted documents."""
    origin, routes, requests = image_server
    routes["/old.gif"] = [(404, {"Content-Type": "text/html"}, b"Not found", 0)]
    routes["/new.gif"] = [(200, {"Content-Type": "image/gif"}, b"GIF89a", 0)]
    source = tmp_path / "docs/source"
    source.mkdir(parents=True)
    (source / "unchanged.rst").write_text(f".. image:: {origin}/old.gif\n")
    deleted = source / "deleted.rst"
    deleted.write_text(f".. figure:: {origin}/old.gif\n")

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "Add existing docs")
    deleted.unlink()
    (source / "new.rst").write_text(f".. figure:: {origin}/new.gif\n")
    git("add", ".")
    git("commit", "-qm", "Update docs")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("check_media.py")), "--base-ref", "HEAD^", "--attempts", "1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "Checked 1 remote images; 0 unavailable" in result.stdout
    assert requests == ["/new.gif"]
