#!/usr/bin/env python3
"""recommend.py — pick the best install combo for the user's system + intent.

Reads preflight.json (or runs preflight internally), gathers user preferences,
filters combos.py by hard requirements, scores survivors against the use case
and env-manager preference, and outputs the chosen combo id along with
ruled-out alternatives and rationale.

Usage:
    python3 scripts/recommend.py                 # interactive
    python3 scripts/recommend.py --non-interactive --use-case rl_research \
        --env-manager uv --isaacsim-source pip --output recommendation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    ask_choice, colorize, confirm, load_combos, print_header, print_info,
    print_ok, print_warn, version_ge,
)

import preflight  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _check_requires(combo, facts):
    """Return (passes, reasons-list). reasons-list is non-empty if filtered out."""
    reqs = combo.get("requires", {})
    reasons = []

    arch = facts["os"].get("machine")
    if "supported_arches" in reqs and arch not in reqs["supported_arches"]:
        reasons.append(f"arch {arch} not in supported set {reqs['supported_arches']}")

    glibc = facts.get("glibc")
    if "min_glibc" in reqs:
        if glibc is None:
            reasons.append("could not detect GLIBC")
        elif not version_ge(glibc, reqs["min_glibc"]):
            reasons.append(f"GLIBC {glibc} < required {reqs['min_glibc']} (pip Isaac Sim needs newer libc)")

    driver = facts["gpu"].get("driver")
    if "min_driver" in reqs:
        if not facts["gpu"]["available"]:
            reasons.append("no NVIDIA GPU / driver detected")
        elif driver and not version_ge(driver, reqs["min_driver"]):
            reasons.append(f"driver {driver} < recommended {reqs['min_driver']}")

    ram = facts.get("ram_gb")
    if "min_ram_gb" in reqs and ram is not None and ram < reqs["min_ram_gb"]:
        reasons.append(f"RAM {ram:.0f}G < recommended {reqs['min_ram_gb']}G")

    disk = facts["disk_free_gb"].get(facts["home"]) or facts["disk_free_gb"].get("/")
    if "min_disk_gb" in reqs and disk is not None and disk < reqs["min_disk_gb"]:
        reasons.append(f"disk free {disk:.0f}G < required {reqs['min_disk_gb']}G")

    tools = reqs.get("requires_tools", [])
    if "conda" in tools and "conda" not in facts["env_managers"] and "mamba" not in facts["env_managers"]:
        reasons.append("conda not installed")
    if "python3.12" in tools and "3.12" not in facts["python_interpreters"]:
        # uv can fetch 3.12; venv combos cannot.
        if combo["env_manager"] == "venv":
            reasons.append("python3.12 not installed (needed for venv combo)")

    if reqs.get("ubuntu_min"):
        v = facts["os"].get("distro_version") or ""
        if facts["os"].get("distro_id") == "ubuntu" and v and not version_ge(v, reqs["ubuntu_min"]):
            reasons.append(f"Ubuntu {v} < required {reqs['ubuntu_min']}")

    return (len(reasons) == 0, reasons)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(combo, prefs, facts):
    """Higher = better fit. Used to break ties among combos that all pass requires."""
    score = 0

    if prefs.use_case in combo.get("recommended_for", []):
        score += 50

    if prefs.env_manager and combo["env_manager"] == prefs.env_manager:
        score += 30
    elif prefs.env_manager == "any":
        # mild preference for uv (matches docs' recommendation)
        if combo["env_manager"] == "uv":
            score += 5

    if prefs.isaacsim_source and combo["isaacsim_source"] == prefs.isaacsim_source:
        score += 25
    elif prefs.isaacsim_source == "any":
        # gentle nudge toward the pip path (docs' recommendation)
        if combo["isaacsim_source"] == "pip":
            score += 5

    if prefs.isaaclab_source and combo["isaaclab_source"] == prefs.isaaclab_source:
        score += 15

    # Penalize advanced unless requested
    if combo["difficulty"] == "advanced" and prefs.use_case != "contribute_isaacsim":
        score -= 20

    # Mild boost to combos that match installed tooling (lower friction)
    if combo["env_manager"] == "uv" and "uv" in facts["env_managers"]:
        score += 3
    if combo["env_manager"] == "conda" and ("conda" in facts["env_managers"] or "mamba" in facts["env_managers"]):
        score += 3

    return score


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

USE_CASE_CHOICES = [
    ("rl_research", "RL training / research (mostly headless)"),
    ("manipulation", "Manipulation, teleop, or vision tasks (may need rendering)"),
    ("sim2real", "Sim-to-real transfer"),
    ("contribute_isaaclab", "Modify Isaac Lab code (contributor)"),
    ("contribute_isaacsim", "Modify Isaac Sim source"),
    ("external_extension", "Build a separate extension/package on top of Isaac Lab"),
    ("kitless_only", "Just Newton physics; no Isaac Sim features needed"),
    ("explore", "Just exploring / not sure yet"),
]

ENV_MANAGER_CHOICES = [
    ("uv", "uv (fastest, recommended)"),
    ("conda", "conda / mamba"),
    ("venv", "stdlib venv"),
    ("any", "no preference — pick the best fit"),
]

ISAACSIM_SOURCE_CHOICES = [
    ("pip", "pip install (easiest, requires GLIBC 2.35+)"),
    ("binary", "downloaded binary zip (works on older Linux)"),
    ("source", "build Isaac Sim from source (only for Isaac Sim contributors)"),
    ("kitless", "no Isaac Sim — Newton physics only"),
    ("any", "no preference — pick the best fit"),
]


class Prefs:
    def __init__(self, use_case, env_manager, isaacsim_source, isaaclab_source, env_name):
        self.use_case = use_case
        self.env_manager = env_manager
        self.isaacsim_source = isaacsim_source
        self.isaaclab_source = isaaclab_source
        self.env_name = env_name


def interactive_prefs(facts):
    print_header("Tell me about your install")
    print_info("(Press Enter to accept defaults shown in [brackets].)")
    print()
    use_case = ask_choice("What's your use case?", USE_CASE_CHOICES, default="rl_research")
    env_manager = ask_choice("Preferred Python environment manager?",
                             ENV_MANAGER_CHOICES, default="uv")
    isaacsim_source = ask_choice("How should Isaac Sim be installed?",
                                 ISAACSIM_SOURCE_CHOICES, default="any")
    # Isaac Lab source rarely needs asking — it's tied to use case
    isaaclab_source = "pip" if use_case == "external_extension" else "source"
    env_name = "env_isaaclab"
    return Prefs(use_case, env_manager, isaacsim_source, isaaclab_source, env_name)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def recommend(facts, prefs):
    combos_mod = load_combos()
    candidates = combos_mod.COMBOS

    passing = []
    rejected = []
    for c in candidates:
        ok, reasons = _check_requires(c, facts)
        if ok:
            passing.append(c)
        else:
            rejected.append((c, reasons))

    # Honor explicit user pick for isaacsim_source / env_manager unless impossible
    def _strict_match(c):
        if prefs.isaacsim_source not in (None, "any") and c["isaacsim_source"] != prefs.isaacsim_source:
            return False
        if prefs.env_manager not in (None, "any") and c["env_manager"] != prefs.env_manager:
            return False
        if prefs.isaaclab_source and c["isaaclab_source"] != prefs.isaaclab_source:
            return False
        return True

    strict = [c for c in passing if _strict_match(c)]
    if strict:
        pool = strict
    else:
        pool = passing

    if not pool:
        return {
            "chosen": None,
            "reason": "No combo satisfies the user's strict preferences and the system requirements.",
            "passing": [c["id"] for c in passing],
            "rejected": [{"id": c["id"], "reasons": rs} for c, rs in rejected],
        }

    scored = sorted(pool, key=lambda c: _score(c, prefs, facts), reverse=True)
    chosen = scored[0]
    rationale = []
    if prefs.use_case in chosen.get("recommended_for", []):
        rationale.append(f"combo is recommended for use case '{prefs.use_case}'")
    if prefs.env_manager not in (None, "any") and chosen["env_manager"] == prefs.env_manager:
        rationale.append(f"matches preferred env manager '{prefs.env_manager}'")
    if prefs.isaacsim_source not in (None, "any") and chosen["isaacsim_source"] == prefs.isaacsim_source:
        rationale.append(f"matches preferred Isaac Sim source '{prefs.isaacsim_source}'")
    if chosen["env_manager"] == "uv" and "uv" in facts["env_managers"]:
        rationale.append("uv already installed on system")

    return {
        "chosen": chosen["id"],
        "title": chosen["title"],
        "difficulty": chosen["difficulty"],
        "rationale": rationale,
        "alternates": [c["id"] for c in scored[1:4]],
        "rejected": [{"id": c["id"], "reasons": rs} for c, rs in rejected],
        "notes": chosen.get("notes", []),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Recommend the best install combo.")
    p.add_argument("--preflight", help="Path to a preflight.json file. If omitted, preflight runs now.")
    p.add_argument("--non-interactive", action="store_true")
    p.add_argument("--use-case", choices=[c[0] for c in USE_CASE_CHOICES])
    p.add_argument("--env-manager", choices=[c[0] for c in ENV_MANAGER_CHOICES])
    p.add_argument("--isaacsim-source", choices=[c[0] for c in ISAACSIM_SOURCE_CHOICES])
    p.add_argument("--isaaclab-source", choices=["source", "pip"])
    p.add_argument("--env-name", default="env_isaaclab")
    p.add_argument("-o", "--output", help="Write recommendation JSON to this path.")
    args = p.parse_args(argv)

    if args.preflight:
        facts = json.loads(Path(args.preflight).read_text())
    else:
        facts = preflight.run_preflight()
        if not args.non_interactive:
            preflight.print_summary(facts, sys.stderr)

    if args.non_interactive:
        if not args.use_case:
            print("--use-case is required in --non-interactive mode", file=sys.stderr)
            return 2
        prefs = Prefs(
            use_case=args.use_case,
            env_manager=args.env_manager or "any",
            isaacsim_source=args.isaacsim_source or "any",
            isaaclab_source=args.isaaclab_source or ("pip" if args.use_case == "external_extension" else "source"),
            env_name=args.env_name,
        )
    else:
        prefs = interactive_prefs(facts)

    rec = recommend(facts, prefs)
    rec["prefs"] = {
        "use_case": prefs.use_case,
        "env_manager": prefs.env_manager,
        "isaacsim_source": prefs.isaacsim_source,
        "isaaclab_source": prefs.isaaclab_source,
        "env_name": prefs.env_name,
    }

    if not args.non_interactive:
        print_header("Recommendation")
        if rec["chosen"]:
            print_ok(f"Chosen combo: {colorize(rec['chosen'], 'cyan')}")
            print_info(rec["title"])
            print_info(f"Difficulty: {rec['difficulty']}")
            if rec["rationale"]:
                print()
                print("Why:")
                for r in rec["rationale"]:
                    print(f"  - {r}")
            if rec["alternates"]:
                print()
                print("Alternates considered:")
                for a in rec["alternates"]:
                    print(f"  - {a}")
            if rec["notes"]:
                print()
                print("Caveats for this combo:")
                for n in rec["notes"]:
                    print(f"  - {n}")
            if rec["rejected"]:
                print()
                print("Combos ruled out by system requirements:")
                for r in rec["rejected"][:6]:
                    print(f"  - {r['id']}: {', '.join(r['reasons'])}")
        else:
            print_warn("No combo passed all hard requirements + your strict preferences.")
            print_info(rec["reason"])
            print()
            print("Combos ruled out by system requirements:")
            for r in rec["rejected"]:
                print(f"  - {r['id']}: {', '.join(r['reasons'])}")

    out = json.dumps(rec, indent=2)
    if args.output:
        Path(args.output).write_text(out)
        if not args.non_interactive:
            print(f"\n[recommend] wrote {args.output}", file=sys.stderr)
    else:
        print(out)
    return 0 if rec["chosen"] else 3


if __name__ == "__main__":
    sys.exit(main())
