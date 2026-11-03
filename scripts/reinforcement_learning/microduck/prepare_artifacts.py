# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Download pinned public teachers to a private cache; do not package checkpoint weights."""

import argparse
import json

from isaaclab_tasks.contrib.microduck.velstand.agents.artifacts import ARTIFACTS, REVISION, resolve_artifact

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cache-dir", default=None)
args = parser.parse_args()
paths = {name: str(resolve_artifact(name, args.cache_dir)) for name in ARTIFACTS}
print(json.dumps({"revision": REVISION, "paths": paths, "sha256": {k: v[1] for k, v in ARTIFACTS.items()}}, indent=2))
