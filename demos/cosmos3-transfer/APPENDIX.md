# Appendix

Everything beyond the Shadow Hand quick start.

## How it works

The camera's `modifiers` is a dict. Each key is one camera output (such as `rgb` or `distance_to_image_plane`) and its
value is a list of modifiers run in order on that output: a chain. The last modifier that sets `output` names the result
published in the camera's data; two chains cannot publish the same name. `presets=cosmos` adds a Cosmos chain that sends
depth to a Cosmos service (over a socket or CUDA IPC) and publishes the generated image as `rgb`. The camera's capture
period sets the gap between frames Cosmos gets; keep it at or under 1/30 s, or the modifier warns. More in the
[integration overview](../../source/isaaclab_experimental/docs/cosmos.md) and
[server setup](../../source/isaaclab_experimental/docs/cosmos_service.md).

## Options

| Option | Meaning |
| --- | --- |
| `--sim-gpu N`, `--cosmos-gpu M` | GPU numbers as `nvidia-smi` lists them (defaults 0 and 1). An inherited `CUDA_VISIBLE_DEVICES` is ignored. |
| `--no-compile` | Eager inference: less memory, slower |
| `--history N` | Cosmos history window (default 30; lower values ghosted in our tests) |
| `--output DIR` | Output folder (default `../output/shadow-<timestamp>`) |

`setup.sh --versions` lists every repository and model snapshot used. The Cosmos Framework is the public repository; this route has not been run on a clean machine.

## ARM (aarch64)

This quick start ran from a clean clone on one ARM workstation (RTX PRO 4000 for the simulator, GB300 for Cosmos) and on
one x86 machine, with `Camera modifier path: PASS` on both. No other ARM system was tested.

## Known limits

- The agent is random, so this shows the integration works, not a policy benefit or a success rate.
- Cosmos adds geometry the simulator does not have (for example below the hand).
- Tested layouts: two 48 GB MIG slices (x86) and an RTX PRO 4000 with a GB300 (aarch64). One shared GPU, or a Cosmos
  GPU under about 40 GB, is not tested.
- The recording script relies on two private details of the camera and modifier (`_data` and `_stream`); a change to
  those classes would break it.

## Testing this branch on a fresh machine (for a person or an agent)

Follow the quick start exactly, with no workarounds and no source changes. A missing token, license or repository access
is not a failure: ask the person, continue with the steps that do not depend on it, and resume. Never accept the Isaac
Sim license for the person. Report, for every command, pass or fail, the exact error, the time and anything done by
hand; then the commit (`git log -1`), the output folder, the PASS or FAIL line, peak GPU memory, and the output of
`./demos/cosmos3-transfer/setup.sh --versions`. Say clearly what you could not test.
