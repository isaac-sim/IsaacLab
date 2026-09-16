# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapt already measured runtime bundles to ASV results and comparison tables.

ASV owns sample statistics, significance testing, and relative comparisons. The
simulator remains in the existing benchmark container; ASV runs on the CI host.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from asv import util
from asv.benchmarks import Benchmarks
from asv.commands.compare import Compare
from asv.config import Config
from asv.results import Results
from asv.runner import BenchmarkResult

from .metrics import METRICS, PerfSmokeError


class AsvComparison:
    """Persist two sample sets in ASV's native format and compare individual metrics."""

    def __init__(
        self,
        directory: Path,
        contract_hash: str,
        history: list[dict[str, float]],
        measurements: list[dict[str, float]],
    ) -> None:
        self.config = Config()
        self.config.results_dir = str(directory)
        self.benchmarks = {
            metric.name: {
                "name": metric.name,
                "version": "1",
                "params": [],
                "param_names": [],
                "type": "track",
                "unit": "seconds" if metric.name in ("total_fps", "startup_time_s") else "GB",
            }
            for metric in METRICS
        }
        Benchmarks(self.config, self.benchmarks.values()).save()
        util.write_json(str(directory / contract_hash / "machine.json"), {"machine": contract_hash}, 1)
        self.results: list[Results] = []
        for name, rows in (("baseline", history), ("candidate", measurements)):
            # These names describe a rolling history and a CI run, not two Git commits.
            result = Results({"machine": contract_hash}, {}, name, 0, "", "runtime", {})
            for metric in METRICS:
                samples = [row[metric.name] for row in rows if metric.name in row]
                if not metric.higher_is_worse:
                    # ASV compares smaller-is-better values. Preserve FPS policy by
                    # converting both the samples and the percentage factor.
                    if any(value <= 0 for value in samples):
                        samples = []
                    else:
                        samples = [1.0 / value for value in samples]
                result.add_result(
                    self.benchmarks[metric.name],
                    BenchmarkResult([0.0 if samples else None], [samples], [1], 0, "", None),
                    record_samples=True,
                )
            result.save(str(directory))
            self.results.append(result)

    def compare(self, metric_name: str, regression_pct: float, higher_is_worse: bool) -> tuple[bool, str]:
        """Return ASV's regression decision and Markdown comparison table."""
        if not 0 <= regression_pct < 100:
            raise PerfSmokeError("ASV regression percentages must be in [0, 100)")
        factor = 1 + regression_pct / 100 if higher_is_worse else 1 / (1 - regression_pct / 100)
        datasets = []
        for result in self.results:
            datasets.append(
                [
                    (
                        metric_name,
                        [],
                        result.get_result_value(metric_name, []),
                        result.get_result_stats(metric_name, []),
                        result.get_result_samples(metric_name, []),
                        "1",
                        result.params["machine"],
                        result.env_name,
                    )
                ]
            )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            worsened, _ = Compare.print_table(
                self.config,
                "baseline",
                "candidate",
                factor=factor,
                split=False,
                resultset_1=datasets[0],
                resultset_2=datasets[1],
                use_stats=True,
            )
        return worsened, output.getvalue()
