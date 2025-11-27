# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import os
import random
import socket
import yaml
from collections import OrderedDict
from pathlib import Path
from prettytable import PrettyTable

from rl_games.algos_torch.torch_ext import safe_filesystem_op, safe_save


class DistributedArgs:
    def __init__(self, args_cli):
        self.distributed = args_cli.distributed
        self.nproc_per_node = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.nnodes = 1
        self.master_port = getattr(args_cli, "master_port", None)

    def get_args_list(self) -> list[str]:
        args = ["-m", "torch.distributed.run", f"--nnodes={self.nnodes}", f"--nproc_per_node={self.nproc_per_node}"]
        if self.master_port:
            args.append(f"--master_port={self.master_port}")
        return args


class EnvArgs:
    def __init__(self, args_cli):
        self.task = args_cli.task
        self.seed = args_cli.seed if args_cli.seed is not None else -1
        self.headless = args_cli.headless
        self.num_envs = args_cli.num_envs

    def get_args_list(self) -> list[str]:
        list = []
        list.append(f"--task={self.task}")
        list.append(f"--seed={self.seed}")
        list.append(f"--num_envs={self.num_envs}")
        if self.headless:
            list.append("--headless")
        return list


class RenderingArgs:
    def __init__(self, args_cli):
        self.camera_enabled = args_cli.enable_cameras
        self.video = args_cli.video
        self.video_length = args_cli.video_length
        self.video_interval = args_cli.video_interval

    def get_args_list(self) -> list[str]:
        args = []
        if self.camera_enabled:
            args.append("--enable_cameras")
        if self.video:
            args.extend(["--video", f"--video_length={self.video_length}", f"--video_interval={self.video_interval}"])
        return args


class WandbArgs:
    def __init__(self, args_cli):
        self.enabled = args_cli.track
        self.project_name = args_cli.wandb_project_name
        self.name = args_cli.wandb_name
        self.entity = args_cli.wandb_entity

    def get_args_list(self) -> list[str]:
        args = []
        if self.enabled:
            args.append("--track")
            if self.entity:
                args.append(f"--wandb-entity={self.entity}")
            else:
                raise ValueError("entity must be specified if wandb is enabled")
            if self.project_name:
                args.append(f"--wandb-project-name={self.project_name}")
            if self.name:
                args.append(f"--wandb-name={self.name}")
        return args


def dump_env_sizes():
    """Print summary of environment variable usage (count, bytes, top-5 largest, SC_ARG_MAX)."""

    n = len(os.environ)
    # total bytes in "KEY=VAL\0" for all envp entries
    total = sum(len(k) + 1 + len(v) + 1 for k, v in os.environ.items())
    # find the 5 largest values
    biggest = sorted(os.environ.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]

    print(f"[ENV MONITOR] vars={n}, total_bytes={total}")
    for k, v in biggest:
        print(f"    {k!r} length={len(v)} → {v[:60]}{'…' if len(v) > 60 else ''}")

    try:
        argmax = os.sysconf("SC_ARG_MAX")
        print(f"[ENV MONITOR] SC_ARG_MAX = {argmax}")
    except (ValueError, AttributeError):
        pass


def flatten_dict(d, prefix="", separator="."):
    """Flatten nested dictionaries into a flat dict with keys joined by `separator`."""

    res = dict()
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            res.update(flatten_dict(value, prefix + key + separator, separator))
        else:
            res[prefix + key] = value

    return res


def find_free_port(max_tries: int = 20) -> int:
    """Return an OS-assigned free TCP port, with a few retries; fall back to a random high port."""
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", 0))
                return s.getsockname()[1]
            except OSError:
                continue
    return random.randint(20000, 65000)


def filter_params(params, params_to_mutate):
    """Filter `params` to only those in `params_to_mutate`, converting str floats (e.g. '1e-4') to float."""

    def try_float(v):
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return v
        return v

    return {k: try_float(v) for k, v in params.items() if k in params_to_mutate}


def save_pbt_checkpoint(workspace_dir, curr_policy_score, curr_iter, algo, params):
    """Save a PBT checkpoint (.pth and .yaml) with policy state, score, and metadata (rank 0 only)."""
    if int(os.environ.get("RANK", "0")) == 0:
        checkpoint_file = os.path.join(workspace_dir, f"{curr_iter:06d}.pth")
        safe_save(algo.get_full_state_weights(), checkpoint_file)
        pbt_checkpoint_file = os.path.join(workspace_dir, f"{curr_iter:06d}.yaml")

        pbt_checkpoint = {
            "iteration": curr_iter,
            "true_objective": curr_policy_score,
            "frame": algo.frame,
            "params": params,
            "checkpoint": os.path.abspath(checkpoint_file),
            "pbt_checkpoint": os.path.abspath(pbt_checkpoint_file),
            "experiment_name": algo.experiment_name,
        }

        with open(pbt_checkpoint_file, "w") as fobj:
            yaml.dump(pbt_checkpoint, fobj)


def load_pbt_ckpts(workspace_dir, cur_policy_id, num_policies, pbt_iteration) -> dict | None:
    """
    Load the latest available PBT checkpoint for each policy (≤ current iteration).
    Returns a dict mapping policy_idx → checkpoint dict or None. (rank 0 only)
    """
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    checkpoints = dict()
    for policy_idx in range(num_policies):
        checkpoints[policy_idx] = None
        policy_dir = os.path.join(workspace_dir, f"{policy_idx:03d}")

        if not os.path.isdir(policy_dir):
            continue

        pbt_checkpoint_files = sorted([f for f in os.listdir(policy_dir) if f.endswith(".yaml")], reverse=True)
        for pbt_checkpoint_file in pbt_checkpoint_files:
            iteration = int(pbt_checkpoint_file.split(".")[0])

            # current local time
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ctime_ts = os.path.getctime(os.path.join(policy_dir, pbt_checkpoint_file))
            created_str = datetime.datetime.fromtimestamp(ctime_ts).strftime("%Y-%m-%d %H:%M:%S")

            if iteration <= pbt_iteration:
                with open(os.path.join(policy_dir, pbt_checkpoint_file)) as fobj:
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"Policy {cur_policy_id} [{now_str}]: Loading"
                        f" policy-{policy_idx} {pbt_checkpoint_file} (created at {created_str})"
                    )
                    checkpoints[policy_idx] = safe_filesystem_op(yaml.load, fobj, Loader=yaml.FullLoader)
                    break

    return checkpoints


def cleanup(checkpoints: dict[int, dict], policy_dir, keep_back: int = 20, max_yaml: int = 50) -> None:
    """
    Cleanup old checkpoints for the current policy directory (rank 0 only).
    - Delete files older than (oldest iteration - keep_back).
    - Keep at most `max_yaml` latest YAML iterations.
    """
    if int(os.environ.get("RANK", "0")) == 0:
        oldest = min((ckpt["iteration"] if ckpt else 0) for ckpt in checkpoints.values())
        threshold = max(0, oldest - keep_back)
        root = Path(policy_dir)

        # group files by numeric iteration (only *.yaml / *.pth)
        groups: dict[int, list[Path]] = {}
        for p in root.iterdir():
            if p.suffix in (".yaml", ".pth") and p.stem.isdigit():
                groups.setdefault(int(p.stem), []).append(p)

        # 1) drop anything older than threshold
        for it in [i for i in groups if i <= threshold]:
            for p in groups[it]:
                p.unlink(missing_ok=True)
            groups.pop(it, None)

        # 2) cap total YAML checkpoints: keep newest `max_yaml` iters
        yaml_iters = sorted((i for i, ps in groups.items() if any(p.suffix == ".yaml" for p in ps)), reverse=True)
        for it in yaml_iters[max_yaml:]:
            for p in groups.get(it, []):
                p.unlink(missing_ok=True)
            groups.pop(it, None)


class PbtTablePrinter:
    """All PrettyTable-related rendering lives here."""

    def __init__(self, *, float_digits: int = 6, path_maxlen: int = 52):
        self.float_digits = float_digits
        self.path_maxlen = path_maxlen

    # format helpers
    def fmt(self, v):
        return f"{v:.{self.float_digits}g}" if isinstance(v, float) else v

    def short(self, s: str) -> str:
        s = str(s)
        L = self.path_maxlen
        return s if len(s) <= L else s[: L // 2 - 1] + "…" + s[-L // 2 :]

    # tables
    def print_params_table(self, params: dict, header: str = "Parameters"):
        table = PrettyTable(field_names=["Parameter", "Value"])
        table.align["Parameter"] = "l"
        table.align["Value"] = "r"
        for k in sorted(params):
            table.add_row([k, self.fmt(params[k])])
        print(header + ":")
        print(table.get_string())

    def print_ckpt_summary(self, sumry: dict[int, dict | None]):
        t = PrettyTable(["Policy", "Status", "Objective", "Iter", "Frame", "Experiment", "Checkpoint", "YAML"])
        t.align["Policy"] = "r"
        t.align["Status"] = "l"
        t.align["Objective"] = "r"
        t.align["Iter"] = "r"
        t.align["Frame"] = "r"
        t.align["Experiment"] = "l"
        t.align["Checkpoint"] = "l"
        t.align["YAML"] = "l"
        for p in sorted(sumry.keys()):
            c = sumry[p]
            if c is None:
                t.add_row([p, "—", "", "", "", "", "", ""])
            else:
                t.add_row([
                    p,
                    "OK",
                    self.fmt(c.get("true_objective", "")),
                    c.get("iteration", ""),
                    c.get("frame", ""),
                    c.get("experiment_name", ""),
                    self.short(c.get("checkpoint", "")),
                    self.short(c.get("pbt_checkpoint", "")),
                ])
        print(t)

    def print_mutation_diff(self, before: dict, after: dict, *, header: str = "Mutated params (changed only)"):
        t = PrettyTable(["Parameter", "Old", "New"])
        for k in sorted(before):
            if before[k] != after[k]:
                t.add_row([k, self.fmt(before[k]), self.fmt(after[k])])
        print(header + ":")
        print(t if t._rows else "(no changes)")
