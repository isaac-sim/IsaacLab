# Image transfer for camera images

`isaaclab_experimental.image_transfer` connects an application-owned image-generation
model to cameras through tensor processors. It does not load models, download assets, or
select a network transport.

For a ready-made adapter with a separately started Cosmos service, see
[Cosmos camera integration](cosmos.md).

## Camera integration

Add the processors to `CameraCfg.tensor_processors`. The camera requests their inputs
that no processor produces, here `distance_to_image_plane`, from its renderer, runs the
processors once per captured image, resets and closes them with the camera, and
publishes the generated images under `"rgb"`:

```python
from isaaclab_experimental.image_transfer import FunctionProcessorCfg, ImageTransferProcessorCfg, depth_to_control

wrist_camera = CameraCfg(
    ...,
    data_types=["rgb"],
    tensor_processors=[
        FunctionProcessorCfg(
            function=depth_to_control,
            params={"near": 0.2, "far": 12.0},
            inputs=["distance_to_image_plane"],
            outputs=["control"],
        ),
        ImageTransferProcessorCfg(backend=MyModelCfg(), initial_frames=1, update_frames=4),
    ],
)
```

Observation terms then read `camera.data.output["rgb"]` as usual. Repeated reads
of one capture do not advance the model.

## Model contract

`ImageTransferProcessorCfg.backend` is a `BackendCfg` whose `class_type` builds an
`ImageTransferModel`. Equal configurations share one model through
`SimulationContext.get_or_create_backend`, so the model holds only shared resources
such as weights. Each processor opens its own `ImageTransferStream` with
`open_stream(num_views, seeds)`; the stream holds the temporal state of its views.
Without a simulation context, the processor builds and owns the model.

`ImageTransferStream.step(controls, reset_rows, seeds)` receives one owned uint8
THWC three-channel control sequence per view and returns uint8 sRGB images with
the same sequence lengths and image size. There is no implicit crop or resize.
Calls run on the current Torch stream; returned tensors must be ready on it.
`close()` must be idempotent. The camera closes its processors, and with them
their streams, when it is released.

## Scheduling and resets

`initial_frames` and `update_frames` set the first and subsequent chunk sizes;
both default to one. The output publishes the last generated image and holds it
until the next chunk, so larger chunks add observation age as well as latency.
Every view needs a complete chunk before a chunk is generated, and
`max_pending_frames` bounds each view's queue.

With `update_period > 0`, environments that reset at different times capture on
different steps. The processor reads the camera's `env_mask` input and queues
controls only for the views captured in each call.

Resetting a view clears its queued controls and its visible image. Its next chunk
uses the initial size, and `step` receives the view index and a new deterministic
seed: view `i` starts from `seed + i`, and each reset advances it by the number of
views, modulo `2**31`. The stream must discard all episode state for those views.
An exception or invalid result marks the processor as failed; recreate the camera
instead of retrying against a stream that may already have advanced.

## Control preparation

Controls are prepared by processors earlier in the list. `depth_to_control` maps
metric depth to fixed near-white, far-black controls; missing, nonfinite, and
nonpositive depth becomes black. `FunctionProcessorCfg` applies any such function to
one camera buffer; any processor that produces uint8 `(N, H, W, 3)` controls under the
image transfer processor's first input can take its place.

## Local example and validation

`examples/sensors/depth_image_processor.py` runs the processors on CPU with a small
illustrative tint model. It requires no model or remote service:

```bash
uv run python examples/sensors/depth_image_processor.py --output output/image-transfer-demo.npz
uv run python -m pytest source/isaaclab_experimental/test/sensors/test_image_transfer.py
```
