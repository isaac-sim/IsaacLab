# 3.0 vs 2.3.2 Regression Evidence

This report is generated from downloaded benchmark artifacts. FPS uses the same steady-state convention as the gate: mean `Environment step effective FPS` after excluded warm-up frames.

## Inputs

- `isaaclab_2_3_2`: `perf-output/regression-evidence-28622521773/regression-evidence-28622521773/isaaclab_2_3_2`
- `current_fork`: `perf-output/regression-evidence-28622521773/regression-evidence-28622521773/current_fork`

## Summary

| Label | Task | Backend | Env count | Samples | Mean FPS | Median FPS | Run-to-run std | Avg within-run std | Mean VRAM MB | Peak VRAM MB | Mean system RAM MB | Peak system RAM MB | CPU | GPU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| current_fork | Isaac-Cartpole-Direct | default | 4096 | 4 | 298620.4 | 301115.8 | 70002.1 | 31449.4 | 1296.6 | 3337.0 | 1806.6 | 4550.7 | INTEL(R) XEON(R) GOLD 5512U (16 physical cores) | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| current_fork | Isaac-Factory-GearMesh-Direct | default | 512 | 4 | 1193.0 | 1207.0 | 34.8 | 72.9 | 4260.9 | 5769.0 | 4522.5 | 8667.1 | INTEL(R) XEON(R) GOLD 5512U (16 physical cores) | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| current_fork | Isaac-Repose-Cube-Shadow-Vision-Benchmark-Direct-v0 | default | 16 | 4 | 331.9 | 332.3 | 12.0 | 32.6 | 1837.0 | 6889.0 | 9237.6 | 15646.7 | INTEL(R) XEON(R) GOLD 5512U (16 physical cores) | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| current_fork | Isaac-Velocity-Flat-G1 | default | 512 | 4 | 10122.0 | 10665.2 | 1264.4 | 669.5 | 1658.5 | 3183.0 | 2257.8 | 4856.8 | INTEL(R) XEON(R) GOLD 5512U (16 physical cores) | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| isaaclab_2_3_2 | Isaac-Cartpole-Direct | physx | 4096 | 4 | 418426.4 | 448289.8 | 78550.6 | 55164.9 | 1273.1 | 3434.0 | 1185.0 | 3718.1 | N/A | N/A |
| isaaclab_2_3_2 | Isaac-Factory-GearMesh-Direct | physx | 512 | 4 | 1251.8 | 1281.6 | 64.0 | 72.1 | 4452.4 | 5932.0 | 9678.4 | 15339.5 | N/A | N/A |
| isaaclab_2_3_2 | Isaac-Velocity-Flat-G1 | physx | 512 | 4 | 11184.4 | 12146.1 | 2073.3 | 661.4 | 1965.3 | 3168.0 | 1704.4 | 4359.2 | N/A | N/A |

## Notes

- `Run-to-run std` is computed across repeated benchmark samples for the same label/task/backend.
- `Avg within-run std` is computed from per-frame steady-state FPS inside each sample, then averaged across samples.
- `Mean VRAM MB` and `Peak VRAM MB` are populated from `nvidia-smi` samples when available. If an artifact only reports benchmark-level GPU memory, that value is used as peak VRAM.
- `Mean system RAM MB` and `Peak system RAM MB` are populated from `docker stats` samples emitted by the evidence workflow.
- Nsight Systems traces are uploaded separately as workflow artifacts and should be copied to Google Drive manually.
