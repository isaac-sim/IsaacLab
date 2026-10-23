# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check remote RST images, including response content, with bounded publication retries."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def collect_images(paths: list[Path]) -> dict[str, tuple[Path, int]]:
    """Collect unique remote figure/image URLs with their first source location."""
    images = {}
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = re.match(r"\s*\.\.\s+(?:figure|image)::\s+(https?://\S+)\s*$", line)
            if match:
                images.setdefault(match[1], (path, line_number))
    return images


def check_image(url: str, *, attempts: int = 3, retry_delay: float = 2, timeout: float = 10) -> str | None:
    """Return a failure reason, or None when GET returns an image type and matching signature.

    Recheck failures, including 404, because externally published assets may still be syncing.
    Read only the first 512 bytes; this checks the format signature, not complete image decoding.
    """
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "IsaacLab-docs-media-check"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get_content_type()
                prefix = response.read(512)
                if response.status != 200:
                    reason = f"HTTP {response.status}"
                elif (
                    content_type == "image/gif"
                    and prefix[:6] in (b"GIF87a", b"GIF89a")
                    or content_type == "image/png"
                    and prefix.startswith(b"\x89PNG\r\n\x1a\n")
                    or content_type == "image/jpeg"
                    and prefix.startswith(b"\xff\xd8\xff")
                    or content_type == "image/webp"
                    and prefix[:4] == b"RIFF"
                    and prefix[8:12] == b"WEBP"
                    or content_type == "image/svg+xml"
                    and re.match(rb"\s*(?:<\?xml.*?\?>\s*)?(?:<!--.*?-->\s*)*<svg(?:\s|>)", prefix, re.DOTALL)
                ):
                    return None
                else:
                    reason = f"response is not a supported image (Content-Type: {content_type})"
        except urllib.error.HTTPError as error:
            reason = f"HTTP {error.code}"
            error.close()
        except (urllib.error.URLError, OSError) as error:
            reason = f"request failed: {error}"
        if attempt + 1 < attempts:
            time.sleep(retry_delay)
    return reason


def main() -> int:
    """Report source-anchored failures; PR callers may request nonblocking reminders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*", help="RST files; defaults to docs/source/**/*.rst")
    parser.add_argument("--base-ref", help="Check only RST files changed from this Git ref to HEAD")
    parser.add_argument("--warn-only", action="store_true", help="Report failures without failing the command")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    if args.attempts < 1 or args.retry_delay < 0 or args.timeout <= 0:
        parser.error("attempts must be positive, retry-delay nonnegative, and timeout positive")
    if args.paths and args.base_ref:
        parser.error("paths and --base-ref are mutually exclusive")

    if args.base_ref:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", args.base_ref, "HEAD", "--", "docs/source"],
            text=True,
        )
        paths = [Path(path) for path in changed.splitlines() if path.endswith(".rst")]
    else:
        paths = args.paths or sorted(Path("docs/source").rglob("*.rst"))
    images = collect_images(paths)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checks = {
            url: pool.submit(
                check_image, url, attempts=args.attempts, retry_delay=args.retry_delay, timeout=args.timeout
            )
            for url in images
        }
        for url, future in checks.items():
            reason = future.result()
            if reason is None:
                continue
            failures += 1
            path, line = images[url]
            message = f"{url}: {reason}. Recheck after any pending upload/sync."
            # Escape workflow-command data so URL/path punctuation cannot corrupt the annotation.
            message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            location = str(path).replace("%", "%25").replace(",", "%2C").replace(":", "%3A")
            level = "warning" if args.warn_only else "error"
            print(f"::{level} file={location},line={line}::{message}", flush=True)
    print(f"Checked {len(images)} remote images; {failures} unavailable or invalid.")
    return int(failures > 0 and not args.warn_only)


if __name__ == "__main__":
    raise SystemExit(main())
