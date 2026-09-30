# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bake a BAM JSON fit into a robot's USD actuator prims before deployment.

The JSON must include firmware constants (error_gain, max_pwm, kp, vin) in addition
to the identified motor/friction fit. The output USD does not reference the JSON.
"""

import argparse
import json
from pathlib import Path

from pxr import Usd

from isaaclab.actuators import BamActuatorCfg
from isaaclab.sim.schemas.schemas_actuators import author_actuator_prims


def main() -> None:
    """Import an extended BAM fit and export a self-contained USD layer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Robot USD to read.")
    parser.add_argument("--output", required=True, help="Output USD (must differ from input).")
    parser.add_argument("--articulation", required=True, help="Articulation root prim path.")
    parser.add_argument("--joint_names", nargs="+", required=True, help="Joint name regular expressions.")
    parser.add_argument("--params_file", required=True, help="Extended BAM fit JSON.")
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("--output must differ from --input")
    values = json.loads(Path(args.params_file).read_text())
    model = values.pop("model", None)
    if model not in ("m1", "m2", "m5", "m6"):
        parser.error(f"Unsupported BAM model {model!r}; expected m1, m2, m5, or m6")
    # Rotor inertia belongs on the joint; this tool only authors controller coefficients.
    for metadata in ("actuator", "q_offset", "armature"):
        values.pop(metadata, None)
    values["resistance"] = values.pop("R")
    values["kp_fw"] = values.pop("kp")
    if values.get("max_current") is None:
        values["max_current"] = 0.0
    values.update(stribeck=int(model != "m1"), load_dependent=int(model in ("m5", "m6")), quadratic=int(model == "m6"))
    stage = Usd.Stage.Open(args.input)
    author_actuator_prims(
        stage,
        args.articulation,
        {
            "bam": BamActuatorCfg(joint_names_expr=args.joint_names, parameter_overrides=values),
        },
    )
    stage.Export(args.output)


if __name__ == "__main__":
    main()
