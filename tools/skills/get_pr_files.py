# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print one changed filename per line for the GitHub PR in the environment.

Reads GH_TOKEN, REPO, PR_NUMBER, and optionally GITHUB_API_URL from the
environment.  Uses only stdlib so it works on any runner without extra deps.
"""

from __future__ import annotations

import json
import os
import urllib.request


def main() -> None:
    token = os.environ["GH_TOKEN"]
    repo = os.environ["REPO"]
    pr = os.environ["PR_NUMBER"]
    base = os.environ.get("GITHUB_API_URL", "https://api.github.com")

    page = 1
    names: list[str] = []
    while True:
        req = urllib.request.Request(
            f"{base}/repos/{repo}/pulls/{pr}/files?per_page=100&page={page}",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
        )
        batch: list[dict] = json.loads(urllib.request.urlopen(req).read())
        names.extend(f["filename"] for f in batch)
        if len(batch) < 100:
            break
        page += 1

    print("\n".join(names))


if __name__ == "__main__":
    main()
