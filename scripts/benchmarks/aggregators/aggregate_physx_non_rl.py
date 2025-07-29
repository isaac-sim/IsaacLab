# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import numpy as np

from utils import bar_plot, check_info_consistency, load_json_files, rework_data

parser = argparse.ArgumentParser()
# A list of benchmark folders
parser.add_argument("--benchmark_folder", type=str, required=True, help="Path to benchmark folder")
args = parser.parse_args()

benchmark_folder = args.benchmark_folder

aggregated_data = {}
json_data = load_json_files(benchmark_folder)
info = check_info_consistency(json_data)
aggregated_data["PhysX"] = {"Metadata": info}
aggregated_data["PhysX"].update(rework_data(json_data, phases=["startup", "runtime"]))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


with open("physx_golden_values_non_rl.json", "w") as f:
    json.dump(aggregated_data, f, cls=NumpyEncoder)
