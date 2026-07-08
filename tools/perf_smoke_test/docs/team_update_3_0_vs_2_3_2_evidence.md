# IsaacLab 3.0 Fork vs 2.3.2 Evidence Pack

This note compiles the evidence collected for the perf-team follow-up on whether
Isaac Lab 2.x FPS numbers can serve as hard floors, and where the current fork
differs from the public 2.3.2 release.

## Slack-Ready Update

We collected a complete evidence bundle for the 2.3.2 vs current-fork
comparison on the RTX PRO 6000 Blackwell runner. The final artifact includes FPS
repeats, VRAM/system RAM samples, CPU utilization samples, and Nsight Systems
traces for Cartpole, G1, Factory, and Shadow Vision.

Main result: 2.3.2 is still faster on the overhead-sensitive tasks, while
Factory is much closer:


| Task             | Env count | 2.3.2 mean FPS | Current fork mean FPS | Delta |
| ---------------- | --------- | -------------- | --------------------- | ----- |
| Cartpole         | 4096      | 418k           | 299k                  | -29%  |
| G1               | 512       | 11.2k          | 10.1k                 | -10%  |
| Factory GearMesh | 512       | 1.25k          | 1.19k                 | -5%   |
| Shadow Vision    | 16        | N/A            | 332                   | N/A   |


The memory data does not point to memory as the cause of the Cartpole/G1 FPS
gap. Peak VRAM is essentially flat between 2.3.2 and the fork for Cartpole and
G1. The main memory outliers are Factory system RAM, where 2.3.2 is higher, and
Shadow Vision on the fork, which peaks around 6.7 GB VRAM / 15.3 GB RAM.

Nsight traces were collected successfully: 8 `.nsys-rep` traces, 8 sqlite
exports, 22 summary CSVs, and 0 missing-trace markers. Initial trace inspection
points to Cartpole as the cleanest regression case: the fork has substantial
CUDA/runtime activity for a simple overhead-bound task, while Factory remains
dominated by PhysX/contact kernels and is close in FPS.

2.3.2 Shadow Vision is marked as "Segfault during rendering startup" in the
plots. Both the benchmark variant and the older v2 task name crash in the
public 2.3.2 container before writing a benchmark JSON, so we do not have a
valid steady-state 2.3.2 memory/FPS KPI for that task.

## Shareable Artifacts

Charts:

- FPS + memory overview: `tools/perf_smoke_test/docs/physx_fps_across_releases.png`
- Memory KPI chart: `tools/perf_smoke_test/docs/memory_kpis_3_0_vs_2_3_2.png`

FPS and memory overview

Memory KPIs

Tables and generated summaries:

- Memory KPI CSV: `tools/perf_smoke_test/docs/memory_kpis_3_0_vs_2_3_2.csv`
- Full generated evidence table: `tools/perf_smoke_test/docs/3_0_vs_2_3_2_regression_evidence.md`
- Machine-readable summary: `tools/perf_smoke_test/docs/regression_evidence_summary.json`

Raw evidence artifacts:

- Final FPS/memory/nsys run: [https://github.com/NVIDIA-Omniverse/IsaacLab/actions/runs/28622521773](https://github.com/NVIDIA-Omniverse/IsaacLab/actions/runs/28622521773)
- Targeted Shadow Vision v2 task-name check: [https://github.com/NVIDIA-Omniverse/IsaacLab/actions/runs/28627373542](https://github.com/NVIDIA-Omniverse/IsaacLab/actions/runs/28627373542)
- Downloaded final artifact root:
`perf-output/regression-evidence-28622521773/regression-evidence-28622521773`
- Downloaded targeted Shadow Vision artifact root:
`perf-output/regression-evidence-28627373542/regression-evidence-28627373542`

## Config

Hardware for the final evidence run:

- CPU: `INTEL(R) XEON(R) GOLD 5512U`, 16 physical cores
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, ~95 GB VRAM
- CUDA reported by benchmark metadata: 12.8
- Workload: `scripts/benchmarks/benchmark_non_rl.py`
- Mode: headless, PhysX/default backend, seed `42`
- FPS convention: mean `Environment step effective FPS` after warm-up frames
are excluded

Tasks:


| Task          | 2.3.2 task                                            | Current fork task                                     | Env count |
| ------------- | ----------------------------------------------------- | ----------------------------------------------------- | --------- |
| Cartpole      | `Isaac-Cartpole-Direct-v0`                            | `Isaac-Cartpole-Direct`                               | 4096      |
| G1            | `Isaac-Velocity-Flat-G1-v0`                           | `Isaac-Velocity-Flat-G1-v0`                           | 512       |
| Factory       | `Isaac-Factory-GearMesh-Direct-v0`                    | `Isaac-Factory-GearMesh-Direct-v0`                    | 512       |
| Shadow Vision | `Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0` | `Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0` | 16        |


## FPS And Stability


| Task             | 2.3.2 mean FPS | Fork mean FPS | 2.3.2 run-to-run std | Fork run-to-run std |
| ---------------- | -------------- | ------------- | -------------------- | ------------------- |
| Cartpole         | 418,426        | 298,620       | 78,551               | 70,002              |
| G1               | 11,184         | 10,122        | 2,073                | 1,264               |
| Factory GearMesh | 1,252          | 1,193         | 64                   | 35                  |
| Shadow Vision    | N/A            | 332           | N/A                  | 12                  |


Interpretation:

- Cartpole remains the best showcase regression: simple scene, 4096 envs,
large gap, and little evidence that memory capacity is involved.
- G1 is slower on this run, but more sensitive to exact config and runner than
Cartpole.
- Factory is close, which supports the hypothesis that the largest gap is not a
pure physics/contact-kernel issue.

## Memory KPIs

Peak memory is the main capacity-planning KPI. VRAM comes from `nvidia-smi`
samples; system RAM comes from `docker stats`.


| Task             | 2.3.2 peak VRAM | Fork peak VRAM | 2.3.2 peak RAM | Fork peak RAM |
| ---------------- | --------------- | -------------- | -------------- | ------------- |
| Cartpole         | 3.4 GB          | 3.3 GB         | 3.6 GB         | 4.4 GB        |
| G1               | 3.1 GB          | 3.1 GB         | 4.3 GB         | 4.7 GB        |
| Factory GearMesh | 5.8 GB          | 5.6 GB         | 15.0 GB        | 8.5 GB        |
| Shadow Vision    | Segfault        | 6.7 GB         | Segfault       | 15.3 GB       |


Interpretation:

- Cartpole/G1 peak VRAM is effectively unchanged across 2.3.2 and the fork.
- Factory uses slightly less peak VRAM and substantially less peak system RAM
on the fork in this run.
- Shadow Vision current-fork memory is measured, but 2.3.2 has no valid
steady-state comparison because it segfaults during rendering startup.

## Nsight Systems Trace Inventory

Final run: [https://github.com/NVIDIA-Omniverse/IsaacLab/actions/runs/28622521773](https://github.com/NVIDIA-Omniverse/IsaacLab/actions/runs/28622521773)

Collected:

- 8 `.nsys-rep` traces
- 8 `.sqlite` exports
- 22 nsys summary CSVs
- 0 `nsys_missing.txt` files

Trace files:


| Label        | Task          | Trace                                                   |
| ------------ | ------------- | ------------------------------------------------------- |
| current fork | Cartpole      | `current_fork/cartpole/nsys/nsys_trace.nsys-rep`        |
| current fork | Factory       | `current_fork/factory/nsys/nsys_trace.nsys-rep`         |
| current fork | G1            | `current_fork/g1/nsys/nsys_trace.nsys-rep`              |
| current fork | Shadow Vision | `current_fork/shadow_vision/nsys/nsys_trace.nsys-rep`   |
| 2.3.2        | Cartpole      | `isaaclab_2_3_2/cartpole/nsys/nsys_trace.nsys-rep`      |
| 2.3.2        | Factory       | `isaaclab_2_3_2/factory/nsys/nsys_trace.nsys-rep`       |
| 2.3.2        | G1            | `isaaclab_2_3_2/g1/nsys/nsys_trace.nsys-rep`            |
| 2.3.2        | Shadow Vision | `isaaclab_2_3_2/shadow_vision/nsys/nsys_trace.nsys-rep` |


Initial nsys read:

- Cartpole current fork shows substantial CUDA/runtime activity in the profiled
sample: about 5.6 s CUDA runtime/API time, ~102k CUDA runtime calls, ~29k GPU
kernel launches, and ~33k sync events.
- The exported 2.3.2 Cartpole summary shows almost no CUDA/kernel activity,
suggesting the fork path adds per-step CUDA/PyTorch/framework work for an
otherwise simple overhead-bound task.
- Factory is dominated by PhysX/contact kernels on both versions, and the FPS
delta is small.

Caveat: CPU backtrace sampling was disabled by runner/system configuration, so
the traces are useful for CUDA/NVTX/OSRT comparison but do not include full CPU
call stacks.

## Shadow Vision 2.3.2 Crash

The 2.3.2 public container crashes before writing benchmark JSON for both:

- `Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0`
- `Isaac-Repose-Cube-Shadow-Vision-Direct-v0`

Observed in logs:

- `isaaclab.python.headless.rendering.kit` launches.
- Rendering/Replicator components initialize.
- `carb.crashreporter-breakpad.plugin` reports `Crash detected`.
- Crash occurs before any valid steady-state benchmark output is written.

Because of this, the graphs explicitly mark the 2.3.2 Shadow Vision position as
`Segfault during rendering startup` instead of showing a misleading partial
memory bar.

## Recommended Message To The Team

Recommended short response:

```text
We now have the comparison artifacts in a shareable state. The final evidence
run collected FPS repeats, VRAM/system RAM samples, CPU stats, and nsys traces
for Cartpole, G1, Factory, and Shadow Vision.

The main pattern is unchanged: 2.3.2 is faster on overhead-bound tasks
(Cartpole ~418k vs fork ~299k, G1 ~11.2k vs ~10.1k), while Factory is close
(~1.25k vs ~1.19k). Memory does not look like the driver of the Cartpole/G1 FPS
gap: peak VRAM is basically flat for those tasks. The nsys traces point to
Cartpole as the cleanest regression case, with the fork showing extra
CUDA/runtime activity for a simple workload.

For perception, the fork Shadow Vision data is valid (~332 FPS, ~6.7 GB peak
VRAM, ~15.3 GB peak RAM). The public 2.3.2 container segfaults during rendering
startup for the corresponding Shadow Vision task, so the plots call that out
explicitly rather than reporting a misleading partial memory sample.
```
