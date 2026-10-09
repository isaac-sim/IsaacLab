# Serving Cosmos for Isaac Lab

Start the Cosmos service in its own model environment, then run Isaac Lab in a
separate terminal and connect its camera to the service. The server keeps the model
loaded between camera sessions. See [Cosmos camera integration](cosmos.md) for the
Isaac Lab configuration and training commands.

The server uses Cosmos Framework for inference and the standalone `isaaclab-cosmos`
package for the camera-session protocol. Installing this package requires only the
Cosmos directory from the Isaac Lab checkout. The server does not require the
`isaaclab` or `isaaclab_experimental` distributions, Isaac Sim, or a running
simulation. The package includes the shared protocol and client source, but starting
the server does not import the camera client.

Cosmos Framework also provides a native [Ray serving interface](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/faq.md#q-how-do-i-run-online-inference-with-ray).
That service accepts HTTP generation requests and returns output files. The server
below supplies the incremental camera interface used by this integration, over a Unix socket on the same
machine or TCP from another machine.

## Prerequisites and paths

Prepare a Cosmos Framework checkout and a supported CUDA environment using its
[official setup guide](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md).
Keep that model environment separate from the Isaac Lab training environment. The
CUDA 13.0 group used below requires Python 3.13 and Torch 2.10; the Isaac Lab
environment on this branch uses Python 3.12 and Torch 2.12. Framework owns its Torch,
CUDA, and compiled model dependencies.

Download the **Cosmos3-Nano-Sim-Transfer checkpoint** from NVIDIA's
[Hugging Face repository](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer)
using the download command below.

The checkpoint directory must contain the weights and complete model configuration
metadata, including a `config.json` with a `model` section for an unregistered HF
export. Its model classes and configuration aliases must be understood by the
selected Framework revision. Some exports contain causal-model aliases missing
from the public loader; an `Unknown Cosmos target alias` or `Unknown Cosmos type
alias` error means the provider must supply compatible metadata or a matching
Framework revision. Installing the service package does not convert the checkpoint.

In the server terminal, set these variables to your absolute paths. All server
commands below use them; set them again when opening a new terminal, or keep them
in your model environment's shell configuration.

```bash
export ISAAC_LAB_ROOT=/absolute/path/to/IsaacLab
export COSMOS_FRAMEWORK_ROOT=/absolute/path/to/cosmos-framework
export COSMOS_CHECKPOINT=/absolute/path/to/Cosmos3-Nano-Sim-Transfer
```

## Set up the server once

From the Framework checkout, follow its environment or container instructions.
For its documented CUDA 13.0 environment:

```bash
cd "$COSMOS_FRAMEWORK_ROOT"
uv sync --locked --all-extras --group=cu130-train
```

`uv sync` creates the Framework environment and installs its dependencies.
Install **only the Cosmos directory** into that environment:

```bash
uv run --no-sync uv pip install --editable \
  "$ISAAC_LAB_ROOT/source/isaaclab_experimental/isaaclab_experimental/cosmos"
```

`uv run` uses the Framework environment without shell activation. The editable
install provides `isaaclab-cosmos-server` from your checkout; keep the checkout in
place.

Follow Framework's [library-path setup](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/setup.md#environment-variables)
in the server terminal to avoid conflicting host libraries:

```bash
export LD_LIBRARY_PATH=
```

Use `--no-sync` when running the server to preserve the separately installed Cosmos
package. If you run `uv sync` again, repeat the Cosmos install afterward.

## Download the checkpoint

From the Framework checkout, download the complete checkpoint into
`COSMOS_CHECKPOINT`:

```bash
uv run --no-sync hf download nvidia/Cosmos3-Nano-Sim-Transfer \
  --local-dir "$COSMOS_CHECKPOINT"
```

If authentication is required, run `uv run --no-sync hf auth login` with an account
that has access to the model repository.

## Start Cosmos

For an eager inference launch, run this in the server terminal:

```bash
cd "$COSMOS_FRAMEWORK_ROOT"
uv run --no-sync isaaclab-cosmos-server \
  --checkpoint "$COSMOS_CHECKPOINT" \
  --no-compile \
  --warmup
```

Wait for `Cosmos ready at unix:///tmp/isaaclab-cosmos-<uid>.sock`, then leave this terminal running. The
server listens on that Unix socket, which only your user can open; it opens no network port.
The worker loads the model once on `cuda:0` and exposes the endpoint after model
loading and warmup. Isaac Lab can then connect from its own environment.

`--max-episode-frames N` sets the longest episode a camera may request, `1 + 4*k` frames. The default, 201,
is the model's trained horizon; `0` removes the cap. See
[Episode length cap](cosmos.md#episode-length-cap).

`--max-views N` lets one camera session batch the cameras of up to N environments (default 1); each needs GPU
memory for its own generation history, and resetting environments independently needs the compiled path. See
[Several environments](cosmos.md#several-environments).

`--kv-window N` and `--attention-sink M` set the generation history the model attends to, in latent frames
(four video frames each). The defaults, 30 and 3, follow the Sim-Transfer recipe. A shorter window is faster and
needs less memory, especially with several environments, but remembers less of each episode.

`--no-compile` selects eager inference. Omit it to enable the compiled CUDA-graph
path, which can take additional time on its first use. `--warmup` runs a disposable
session on a `(480, 832)` canvas through the full history window (`1 + 4 * --kv-window` frames, 121 by default,
within the episode cap) before reporting readiness. With compiled
inference, other canvases or prompts can still require compilation on their first
use. Omit `--warmup` to expose the endpoint after model loading.

The equivalent module entrypoint is
`uv run --no-sync python -m isaaclab_experimental.cosmos.server.worker` with the same
arguments. Use `--device cuda:1` for another GPU, and `--endpoint unix:///path/to/socket` or
`--endpoint tcp://127.0.0.1:5556` for another endpoint, for example TCP for Isaac Lab on another machine through
an SSH tunnel (see [Endpoints and transports](cosmos.md#endpoints-and-transports)). Match that endpoint in the camera
configuration.

## Connect Isaac Lab

In a second terminal, use the Isaac Lab environment and its checkout:

```bash
cd /absolute/path/to/IsaacLab
uv run isaaclab cosmos status
```

A ready service reports `"ready": true`. Before connecting a camera, check that
`"session_active": false`: the server runs one active generation session, with up
to `--max-views` camera views. If a session is active, close that camera or Isaac Lab
process before connecting another. For another endpoint, pass
the same `--endpoint` to the status command.

Run the camera task or training command from [Cosmos camera integration](cosmos.md).
Resetting or closing the camera clears its generation history while keeping the
model loaded. Closing Isaac Lab leaves the service running for the next command.

## Stop Cosmos

Press Ctrl+C in the Cosmos server terminal to stop the service and release the
model.
