# Tissue and appearance kernels

> Scheduled for replacement by the Newton manager; see "Follow-up work" in the task README.

Derived from the local Carsten POC at
`/mnt/data/isaac_lab_poc/fruit_handling/carsten_experiments/squishy/`.
The original source and prepared assets were not modified.

Retained: explicit APIC/MLS MPM, fixed-corotated elasticity, plasticity,
damage/tensile softening, finite-lived adhesive contact, fracture-aware MLS32
Gaussian binding, covariance deformation, and SH material-frame shading.

Task-specific changes:

- `explicit_mpm.py`: selectable output/contact frequency (`frame_hz`, default
  unchanged at 30 Hz); skip unused interface accumulators in single-field mode.
- `rtx_sh_frame.py`: removed the unused POC viewer installation helper; the task
  authors the same material-frame attributes in its own USD/viewer adapter.
  Optional tint-only identity frames bypass live SH rotation without changing bruising.
- Live measured Franka pad poses, reciprocal impulses, reset, and timestep
  selection are implemented outside these kernels, in `../runtime.py`.

These files are kept separate from the Isaac Lab integration to make upstream
comparison possible. This local reuse does not establish additional distribution
rights for the POC or NVIDIA MDL source. Asset attribution and shader provenance
are included in each asset package.
