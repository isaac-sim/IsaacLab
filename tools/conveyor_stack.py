# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rebuild the conveyor integration draft from the current component PR heads."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def run(args: list[str], cwd: Path, *, env: dict[str, str] | None = None, capture: bool = False) -> str:
    """Run an argument vector and stop immediately on failure."""
    result = subprocess.run(args, cwd=cwd, env=env, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def main() -> None:
    """Fetch, merge, validate, and optionally publish the current stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run focused CPU tests before publishing.")
    parser.add_argument("--push", action="store_true", help="Update the integration branch and draft description.")
    parser.add_argument("--checkpoint", type=Path, help="Also evaluate this trained policy for 360 CPU steps.")
    args = parser.parse_args()
    if args.push and not args.check:
        parser.error("--push requires --check")
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "tools/conveyor_stack.json").read_text())
    repository = manifest["repository"]
    branch = manifest["integration_branch"]
    if branch in {"main", "develop"} or branch.startswith("release/"):
        parser.error("The integration target must be a feature branch.")
    git = ["git", "-c", "http.version=HTTP/1.1"]
    expected = run(git + ["ls-remote", "fork", f"refs/heads/{branch}"], root, capture=True).split()[0]
    components = []
    refs = [f"+refs/heads/{manifest['base']}:refs/conveyor-stack/base"]
    for component in manifest["components"]:
        number = int(component["pr"])
        data = json.loads(run(["gh", "api", f"repos/{repository}/pulls/{number}"], root, capture=True))
        if data["state"] != "open" and not data["merged"]:
            raise RuntimeError(f"PR #{number} was closed without merging; update the component list.")
        components.append({**component, "head": data["head"]["sha"], "merged": data["merged"]})
        if not data["merged"]:
            refs.append(f"+refs/pull/{number}/head:refs/conveyor-stack/{number}")
    run(git + ["fetch", "--no-tags", f"https://github.com/{repository}.git", *refs], root)
    base = run(git + ["rev-parse", "refs/conveyor-stack/base"], root, capture=True)
    with tempfile.TemporaryDirectory(prefix="conveyor-stack-") as directory:
        checkout = Path(directory) / "checkout"
        run(git + ["worktree", "add", "--detach", str(checkout), base], root)
        try:
            for component in components:
                if component["merged"]:
                    continue
                number = component["pr"]
                head = run(git + ["rev-parse", f"refs/conveyor-stack/{number}"], root, capture=True)
                if head != component["head"]:
                    raise RuntimeError(f"PR #{number} changed during fetch; rerun the refresh.")
                run(git + ["merge", "--no-ff", "-m", f"Integrate #{number} at {head[:12]}", head], checkout)
            for name in ("conveyor_stack.py", "conveyor_stack.json", "conveyor_stack.md"):
                shutil.copyfile(root / "tools" / name, checkout / "tools" / name)
            env = os.environ.copy()
            env["UV_PROJECT_ENVIRONMENT"] = str(root / ".venv")
            env["ISAACLAB_TEST_DEVICES"] = "100"
            env["PYTHONPATH"] = os.pathsep.join(str(path) for path in (checkout / "source").iterdir() if path.is_dir())
            uv = ["uv", "run", "--no-sync"]
            checks = []
            if args.check:
                tests = [
                    "source/isaaclab/test/sim/test_spawn_meshes.py",
                    "source/isaaclab_contrib/test/conveyors/test_surface_velocity_spec.py",
                    "source/isaaclab_contrib/test/conveyors/test_surface_velocity_newton.py",
                    "source/isaaclab_contrib/test/conveyors/test_surface_velocity_physx.py",
                    "source/isaaclab_tasks/test/contrib/test_conveyor_franka_geometry.py",
                    "source/isaaclab_tasks/test/contrib/test_conveyor_franka_mdp.py",
                    "source/isaaclab_tasks/test/contrib/test_conveyor_franka_physx_cfg.py",
                    "source/isaaclab_tasks/test/contrib/test_conveyor_franka_asset_cfg.py",
                ]
                for test in tests:
                    run(uv + ["python", "-m", "pytest", test, "-q"], checkout, env=env)
                run(["uv", "lock", "--check"], checkout, env=env)
                checks.append("Focused CPU tests and lock consistency passed")
                if args.checkpoint:
                    report = Path(directory) / "policy.json"
                    run(
                        uv
                        + [
                            "python",
                            "-m",
                            "isaaclab_tasks.contrib.conveyor_franka.evaluate",
                            "--checkpoint",
                            str(args.checkpoint.resolve()),
                            "--num_envs",
                            "2",
                            "--device",
                            "cpu",
                            "--steps",
                            "360",
                            "--seed",
                            "0",
                            "--output",
                            str(report),
                        ],
                        checkout,
                        env=env,
                    )
                    policy = json.loads(report.read_text())
                    if policy["failures"] or policy["nonfinite_observations"] or not policy["completed_transfers"]:
                        raise RuntimeError(f"Policy evaluation did not meet the bounded transfer check: {policy}")
                    checks.append(
                        f"360-step policy evaluation: {policy['completed_transfers']} transfers, "
                        "zero safety resets/non-finite observations"
                    )
            manifest["snapshot"] = {
                "base": base,
                "components": components,
                "checks": checks,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
            (checkout / "tools/conveyor_stack.json").write_text(json.dumps(manifest, indent=2) + "\n")
            run(
                git + ["add", "tools/conveyor_stack.py", "tools/conveyor_stack.json", "tools/conveyor_stack.md"],
                checkout,
            )
            run(uv + ["isaaclab", "-f"], checkout, env=env)
            run(git + ["diff", "--check"], checkout)
            run(git + ["commit", "-m", "Record the tested conveyor integration snapshot"], checkout)
            head = run(git + ["rev-parse", "HEAD"], checkout, capture=True)
            rows = "\n".join(f"| #{item['pr']} | {item['name']} | `{item['head'][:12]}` |" for item in components)
            body = (
                "## Conveyor integration draft\n\n**Do not merge this PR.** "
                "Review and merge the four component PRs individually.\n\n"
                f"| PR | Scope | Included head |\n| --- | --- | --- |\n{rows}\n\n"
                f"Base: `{base[:12]}`. Integration commit: `{head[:12]}`.\n\n"
                + "\n".join(f"- {check}" for check in checks)
                + "\n\nRefresh from the latest published PR heads with:\n\n"
                "```bash\nuv run --no-project python tools/conveyor_stack.py --check --push\n```\n\n"
                "Add `--checkpoint /path/to/model.pt` for bounded policy evaluation. "
                "A failed test or merge conflict stops publication. The refresh uses a temporary checkout "
                "and records exact SHAs in `tools/conveyor_stack.json`. "
                "It runs on demand; it does not poll or refresh automatically.\n"
            )
            if args.push:
                run(
                    git
                    + [
                        "push",
                        f"--force-with-lease=refs/heads/{branch}:{expected}",
                        "fork",
                        f"HEAD:refs/heads/{branch}",
                    ],
                    checkout,
                )
                payload = Path(directory) / "description.json"
                payload.write_text(json.dumps({"body": body}))
                run(
                    [
                        "gh",
                        "api",
                        "--method",
                        "PATCH",
                        f"repos/{repository}/pulls/{manifest['integration_pr']}",
                        "--input",
                        str(payload),
                        "--silent",
                    ],
                    root,
                )
            else:
                print(body)
        finally:
            run(git + ["worktree", "remove", "--force", str(checkout)], root)


if __name__ == "__main__":
    main()
