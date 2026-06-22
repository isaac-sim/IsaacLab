"""Keyboard diagnostic for Isaac Sim / carb.input.

Prints every keyboard event received by carb.input.subscribe_to_keyboard_events.
Use this to verify which keys actually reach the carb subscriber and what
names/values they carry, before debugging the robot control script.

Usage:
    ./isaaclab.sh -p scripts/tools/debug_keyboard.py

Press keys in the Isaac Sim window.
Press ESC or Ctrl-C to exit.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── everything below runs after Kit is up ────────────────────────────────────

import carb.input as ci
import omni.appwindow

_PRESS   = ci.KeyboardEventType.KEY_PRESS
_RELEASE = ci.KeyboardEventType.KEY_RELEASE

def on_event(event, *_) -> bool:
    """Print raw carb keyboard event info."""
    raw = event.input

    try:
        name_attr = raw.name
    except AttributeError:
        name_attr = "<no .name>"

    str_repr = str(raw)

    try:
        int_val = int(raw)
    except (TypeError, ValueError):
        int_val = None

    evt_type = "PRESS  " if event.type == _PRESS else "RELEASE"
    print(
        f"[{evt_type}]  "
        f".name={name_attr!r:25s}  "
        f"str={str_repr!r:35s}  "
        f"int={int_val}"
    )
    return True


appwindow   = omni.appwindow.get_default_app_window()
keyboard    = appwindow.get_keyboard()
input_iface = ci.acquire_input_interface()
sub = input_iface.subscribe_to_keyboard_events(keyboard, on_event)

print("\n=== Keyboard Diagnostic ===")
print("Press keys in the Isaac Sim window.")
print("Useful keys to test: arrows, PgUp, PgDn, [, ], I, O, Z, X, K, R, N, TAB")
print("Press ESC or Ctrl-C to exit.\n")

import contextlib
try:
    while simulation_app.is_running():
        simulation_app.update()
except KeyboardInterrupt:
    pass

input_iface.unsubscribe_to_keyboard_events(keyboard, sub)
simulation_app.close()
