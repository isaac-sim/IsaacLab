# OVRTX GPU-transform golden images

`test_ovrtx_gpu_transforms_golden.py` exercises the native renderer with two simultaneous
640 × 360 cameras, a checkerboard floor, and a moving sphere and cube. It enables
`RendererConfig.read_gpu_transforms`, the native option controlled by Isaac Lab's
`ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS`. This is a GPU integration test and needs an
RTX-capable device; it does not launch Isaac Sim or use remote scene assets.

After 40 warmup frames, the objects move for 60 frames and hold their final poses
for another 40. Each of the final ten RGB frames from both cameras is compared
individually with its golden. Checking both cameras matters because the affected
render product can change between processes. Alpha is excluded. Following the
task-level OVRTX pixel tolerance, at most 3% of pixels may have RGB L2 error above
10 (on the 0–255 scale). A failure saves the worst actual image, reference, pixel
difference mask, and renderer log under pytest's temporary directory.

## Reference provenance

The two PNGs contain the rounded per-channel median of the final ten frames
rendered with **GPU transform reads disabled**. They were generated on 2026-09-16
with an NVIDIA A40, driver 580.173.02, and OVRTX
`0.5.0.377615.868bf616.manylinux_2_35_x86_64`.
SDK archive SHA-256:
`598bb1147b71483e104d236a0ad56b369f42a2bf78111272f60d0a7d74c37e6a`.

The scene is self-contained; the test generates its checkerboard texture beside
the temporary USD file. Goldens and the USD fixture are tracked with Git LFS.
Missing references are errors; the test never creates or replaces them.

Validation on the reference machine (maximum differing-pixel percentage across
both cameras and all ten comparison frames; limit 3%):

| Runtime | GPU transform reads | Result | Maximum difference |
| --- | --- | --- | --- |
| 0.4.1.364340 | Enabled | Failed | 20.14% |
| 0.4.1.364340 | Disabled, control capture | Passed | 1.85% |
| 0.5.0.377615 | Enabled | Passed | 0.52% |
| 0.5.0.377615 | Enabled, independent repeat | Passed | 0.50% |

These tolerances have been validated on this A40; other GPU/driver combinations
may need evaluation before adding this test to a broader golden-image matrix.

## Run

Install the `ovrtx` and `test` project extras, then run this file in its own process:

```bash
uv run --frozen --no-sync python -m pytest \
  source/isaaclab_ov/test/test_ovrtx_gpu_transforms_golden.py -q
```

The project currently pins OVRTX `0.4.1.364340`, which is expected to fail this
regression. To test the extracted 0.5 SDK without changing that environment:

```bash
PYTHONPATH=/path/to/ovrtx-sdk/python uv run --frozen --no-sync python -m pytest \
  source/isaaclab_ov/test/test_ovrtx_gpu_transforms_golden.py -q
```

The assertions do not inspect the version or mark older versions as expected
failures. OVRTX 0.4 and 0.5 use different output dictionary keys; that API
difference is handled only when reading the rendered image.

## Deliberate reference updates

Only regenerate after reviewing the scene or intended appearance change. Use a
verified renderer with GPU reads disabled, visually inspect both images, and
rerun the GPU-on test. The following can be saved as a temporary Python script
and executed with `uv run --frozen --no-sync python` under that renderer:

```python
import runpy
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

test = runpy.run_path("source/isaaclab_ov/test/test_ovrtx_gpu_transforms_golden.py")
with tempfile.TemporaryDirectory() as directory:
    frames, version = test["_capture_frames"](Path(directory), read_gpu_transforms=False)
for camera_index in range(2):
    golden = np.rint(np.median(frames[:, camera_index], axis=0)).astype(np.uint8)
    Image.fromarray(golden).save(test["_GOLDEN_DIR"] / f"camera{camera_index}.png")
print(f"Generated GPU-off references with native OVRTX {version}")
```
