"""Shared utilities for the isaaclab-installer skill.

Pure stdlib only. Compatible with Python 3.6+ so that preflight/recommend can
run on the user's system Python before any environment is created.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SKILL_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = SKILL_ROOT / "resources"
SCRIPTS_DIR = SKILL_ROOT / "scripts"
SMOKE_DIR = RESOURCES_DIR / "smoke_tests"

# Default user-facing artifact locations.
DEFAULT_PROFILE_DIR = Path.home() / ".isaaclab"
DEFAULT_PROFILE_PATH = DEFAULT_PROFILE_DIR / "install_profile.yaml"
DEFAULT_LOG_DIR = DEFAULT_PROFILE_DIR / "logs"

# Sentinel used by recommend.py when no combo fits.
NO_COMBO = "__none__"


# ---------------------------------------------------------------------------
# Importing the combos module without polluting sys.path globally.
# ---------------------------------------------------------------------------

def load_combos():
    """Import and return the resources.combos module."""
    import importlib.util

    combos_path = RESOURCES_DIR / "combos.py"
    spec = importlib.util.spec_from_file_location("isaaclab_installer_combos", combos_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import combos module from {combos_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Subprocess helpers (safe defaults)
# ---------------------------------------------------------------------------

def run(cmd, cwd=None, env=None, check=False, capture=True, timeout=None):
    """Run a shell command. Returns subprocess.CompletedProcess.

    - cmd may be a list or string. If string, shell=True is used.
    - capture=True returns stdout/stderr as strings.
    - never raises on non-zero by default; caller inspects returncode.
    """
    is_shell = isinstance(cmd, str)
    try:
        result = subprocess.run(
            cmd,
            shell=is_shell,
            cwd=cwd,
            env=env,
            check=check,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=True,
            timeout=timeout,
        )
        return result
    except FileNotFoundError as e:
        # Synthesize a "not found" result so callers can keep going.
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(e))
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(cmd, 124, stdout=e.stdout or "", stderr=str(e))


def which(name):
    """shutil.which wrapper returning None if missing."""
    return shutil.which(name)


def get_version(cmd_list, regex, group=1):
    """Run cmd_list, search regex on combined stdout+stderr, return group or None."""
    res = run(cmd_list)
    text = (res.stdout or "") + (res.stderr or "")
    m = re.search(regex, text)
    return m.group(group) if m else None


# ---------------------------------------------------------------------------
# Version comparison helpers
# ---------------------------------------------------------------------------

def _vtuple(v):
    """Parse a dotted version string into a tuple of ints, ignoring trailing junk."""
    if v is None:
        return ()
    parts = []
    for p in re.split(r"[.\-+]", str(v)):
        m = re.match(r"^(\d+)", p)
        if not m:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


def version_ge(a, b):
    """True if version a >= version b (both dotted strings)."""
    if a is None:
        return False
    return _vtuple(a) >= _vtuple(b)


# ---------------------------------------------------------------------------
# Disk / RAM helpers (Linux-focused but won't crash on others)
# ---------------------------------------------------------------------------

def disk_free_gb(path):
    try:
        usage = shutil.disk_usage(str(path))
        return usage.free / (1024 ** 3)
    except OSError:
        return None


def total_ram_gb():
    """Read /proc/meminfo (Linux); fall back to None elsewhere."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Pretty I/O
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"


def _supports_color():
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def colorize(text, color):
    if not _supports_color():
        return text
    codes = {"red": _RED, "green": _GREEN, "yellow": _YELLOW,
             "blue": _BLUE, "cyan": _CYAN, "bold": _BOLD}
    return f"{codes.get(color, '')}{text}{_RESET}"


def print_header(text):
    print()
    print(colorize("=" * 72, "blue"))
    print(colorize(text, "bold"))
    print(colorize("=" * 72, "blue"))


def print_step(text):
    print(colorize("\n--> " + text, "cyan"))


def print_ok(text):
    print(colorize("[OK]  ", "green") + text)


def print_warn(text):
    print(colorize("[WARN]", "yellow") + " " + text)


def print_err(text):
    print(colorize("[ERR] ", "red") + text)


def print_info(text):
    print("      " + text)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    return path


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# YAML-ish writer (for install_profile.yaml) — pure stdlib.
# ---------------------------------------------------------------------------

def write_yaml(path, data):
    """Write a dict/list as a minimal YAML file. Strings are quoted only when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        _emit(f, data, 0)
    return path


def _scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    needs_quotes = (
        s == "" or s != s.strip() or
        s.lower() in ("yes", "no", "true", "false", "null", "~") or
        re.match(r"^-?\d", s) is not None or
        re.search(r"[:#&*!|>%@`,\[\]\{\}]", s) is not None
    )
    if needs_quotes:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def _emit(f, obj, indent):
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            f.write(pad + "{}\n")
            return
        for k, v in obj.items():
            ks = _scalar(k)
            if isinstance(v, (dict, list)) and v:
                f.write(f"{pad}{ks}:\n")
                _emit(f, v, indent + 1)
            else:
                f.write(f"{pad}{ks}: {_scalar(v) if not isinstance(v, (dict, list)) else ('{}' if isinstance(v, dict) else '[]')}\n")
    elif isinstance(obj, list):
        if not obj:
            f.write(pad + "[]\n")
            return
        for item in obj:
            if isinstance(item, (dict, list)) and item:
                f.write(pad + "-\n")
                _emit(f, item, indent + 1)
            else:
                f.write(f"{pad}- {_scalar(item)}\n")
    else:
        f.write(pad + _scalar(obj) + "\n")


def read_yaml(path):
    """Best-effort YAML reader. Uses PyYAML if available, else a tiny parser
    that handles the subset our skill writes (block dicts/lists, simple scalars)."""
    try:
        import yaml  # type: ignore
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except ImportError:
        pass
    return _mini_yaml_load(Path(path).read_text())


def _mini_yaml_load(text):
    """Minimal YAML loader. Supports:
       - dicts: key: value
       - lists: '- item'
       - nested via indentation (2 spaces)
       - scalars: ints, floats, true/false/null/~, strings (quoted or unquoted)
       - comments (#) and blank lines
    Limitations: no flow style, no anchors, no multi-line scalars."""
    lines = []
    for raw in text.splitlines():
        # strip trailing comments outside of quotes (approximate but fine for our writer)
        in_str = None
        cleaned_chars = []
        for ch in raw:
            if in_str:
                cleaned_chars.append(ch)
                if ch == in_str:
                    in_str = None
            else:
                if ch == "#":
                    break
                if ch in ('"', "'"):
                    in_str = ch
                cleaned_chars.append(ch)
        cleaned = "".join(cleaned_chars).rstrip()
        if cleaned.strip() == "":
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        lines.append((indent, cleaned.strip()))

    def parse_block(start, base_indent):
        i = start
        # Detect list vs dict
        if i < len(lines) and lines[i][1].startswith("- "):
            result = []
            while i < len(lines):
                indent, content = lines[i]
                if indent < base_indent:
                    break
                if indent > base_indent:
                    # belongs to previous item; should have been consumed
                    break
                if not content.startswith("- "):
                    break
                rest = content[2:].strip()
                if ": " in rest or rest.endswith(":"):
                    # dict starting on same line as dash
                    item_lines_start = i
                    # Treat the rest as the first kv of a dict
                    sub = {}
                    key, sep, val = rest.partition(":")
                    key = key.strip()
                    val = val.strip()
                    if val:
                        sub[key] = _scalar_load(val)
                        i += 1
                    else:
                        i += 1
                        # consume nested block at indent + 2
                        sub_val, i = parse_block(i, base_indent + 2)
                        sub[key] = sub_val
                    # absorb further keys at indent + 2 belonging to this item
                    while i < len(lines) and lines[i][0] == base_indent + 2 and not lines[i][1].startswith("- "):
                        k_line = lines[i][1]
                        k, _, v = k_line.partition(":")
                        k = k.strip(); v = v.strip()
                        if v:
                            sub[k] = _scalar_load(v)
                            i += 1
                        else:
                            i += 1
                            child, i = parse_block(i, base_indent + 4)
                            sub[k] = child
                    result.append(sub)
                else:
                    result.append(_scalar_load(rest))
                    i += 1
            return result, i

        result = {}
        while i < len(lines):
            indent, content = lines[i]
            if indent < base_indent:
                break
            if indent > base_indent:
                break
            if content.startswith("- "):
                break
            key, sep, val = content.partition(":")
            if not sep:
                # bare scalar inside what we thought was a dict; bail
                break
            key = key.strip()
            val = val.strip()
            if val:
                result[key] = _scalar_load(val)
                i += 1
            else:
                i += 1
                child, i = parse_block(i, base_indent + 2)
                result[key] = child
        return result, i

    data, _ = parse_block(0, 0)
    return data


def _scalar_load(s):
    if s == "" or s == "~" or s.lower() == "null":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def confirm(prompt, default=False, assume_yes=False):
    """Ask a yes/no question. Returns True/False."""
    if assume_yes:
        return True
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        try:
            ans = input(colorize(prompt + suffix, "bold")).strip().lower()
        except EOFError:
            return default
        if ans == "":
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  Please answer y or n.")


def ask(prompt, default=None, validator=None):
    """Free-form input with optional default and validator(value)->(ok, msg)."""
    suffix = f" [{default}]: " if default is not None else ": "
    while True:
        try:
            ans = input(colorize(prompt + suffix, "bold")).strip()
        except EOFError:
            ans = ""
        if ans == "" and default is not None:
            ans = default
        if validator:
            ok, msg = validator(ans)
            if not ok:
                print(colorize("  " + msg, "yellow"))
                continue
        return ans


def ask_choice(prompt, choices, default=None):
    """Ask user to pick one of `choices` (list of strings or (id, label) tuples)."""
    pairs = [(c, c) if isinstance(c, str) else c for c in choices]
    print(colorize(prompt, "bold"))
    for i, (cid, label) in enumerate(pairs, 1):
        marker = " (default)" if default == cid else ""
        print(f"  {i}. {label}{marker}")
    while True:
        try:
            ans = input("  Enter number" + (f" [default {default}]" if default else "") + ": ").strip()
        except EOFError:
            ans = ""
        if ans == "" and default:
            return default
        if ans.isdigit() and 1 <= int(ans) <= len(pairs):
            return pairs[int(ans) - 1][0]
        # match by id too
        for cid, _ in pairs:
            if ans == cid:
                return cid
        print(colorize("  Please enter a number from the list.", "yellow"))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def open_log(name="install"):
    """Open a timestamped log file in DEFAULT_LOG_DIR. Returns (path, file-handle)."""
    from datetime import datetime
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_path = DEFAULT_LOG_DIR / f"{name}-{ts}.log"
    fh = open(log_path, "w")
    return log_path, fh


def now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
