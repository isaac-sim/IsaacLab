# Image transfer for camera images

`isaaclab_contrib.image_transfer` connects an application-owned image-generation
model to cameras through modifiers. It does not load models, download assets, or
select a network transport.

## Camera integration

Add the modifiers to `CameraCfg.modifiers` under the camera output the model is
conditioned on. The camera requests that output from its renderer, runs the chain
once per captured image, resets it with the camera, and publishes the generated
images under `"rgb"`:

```python
from isaaclab.utils.modifiers import ModifierCfg
from isaaclab_contrib.image_transfer import ImageTransferModifierCfg, depth_to_control

wrist_camera = CameraCfg(
    ...,
    data_types=["rgb"],
    modifiers={
        "distance_to_image_plane": [
            ModifierCfg(func=depth_to_control, params={"near": 0.2, "far": 12.0}),
            ImageTransferModifierCfg(backend=MyModelCfg(), initial_frames=1, update_frames=4),
        ]
    },
)
```

Observation terms then read `camera.data.output["rgb"]` as usual. Repeated reads
of one capture do not advance the model.

## Model contract

`ImageTransferModifierCfg.backend` is a `BackendCfg` whose `class_type` builds an
`ImageTransferModel`. Equal configurations share one model through
`SimulationContext.get_or_create_backend`, so the model holds only shared resources
such as weights. Each modifier opens its own `ImageTransferStream` with
`open_stream(num_views, seeds)`; the stream holds the temporal state of its views.
Without a simulation context, the modifier builds and owns the model.

`ImageTransferStream.step(controls, reset_rows, seeds)` receives one owned uint8
THWC three-channel control sequence per view and returns uint8 sRGB images with
the same sequence lengths and image size. There is no implicit crop or resize.
Calls run on the current Torch stream; returned tensors must be ready on it.
`close()` must be idempotent. The camera closes its modifiers, and with them
their streams, when it is released.

## Scheduling and resets

`initial_frames` and `update_frames` set the first and subsequent chunk sizes;
both default to one. The output publishes the last generated image and holds it
until the next chunk, so larger chunks add observation age as well as latency.
Every view needs a complete chunk before a chunk is generated, and
`max_pending_frames` bounds each view's queue.

Resetting a view clears its queued controls and its visible image. Its next chunk
uses the initial size, and `step` receives the view index and a new deterministic
seed: view `i` starts from `seed + i`, and each reset advances it by the number of
views, modulo `2**31`. The stream must discard all episode state for those views.
An exception or invalid result marks the modifier as failed; recreate the camera
instead of retrying against a stream that may already have advanced.

Camera chains process all views of each capture together.

## Control preparation

Controls are prepared by modifiers earlier in the chain. `depth_to_control` maps
metric depth to fixed near-white, far-black controls; missing, nonfinite, and
nonpositive depth becomes black. Any function or class modifier that returns
uint8 `(N, H, W, 3)` controls can take its place, for example one that computes
edges from `"rgb"` or colors a `"semantic_segmentation"` output.

## Optional PPISP

Append `ModifierCfg(func=srgb_to_linear)` and `PpispModifierCfg(isp_cfg=...)` to
the chain to apply relative camera effects to the generated images. The inverse
sRGB transfer function does not recover scene HDR or remove a baked camera
response; use the renderer's `rgb_radiance` for calibrated PPISP processing. The
image transfer modifiers do not depend on PPISP.

## Local example and validation

`examples/sensors/depth_image_processor.py` runs the chain on CPU with a small
illustrative tint model. It requires no model or remote service:

```bash
uv run python examples/sensors/depth_image_processor.py --output output/image-transfer-demo.npz
uv run python -m pytest source/isaaclab_contrib/test/sensors/test_image_transfer.py
```
