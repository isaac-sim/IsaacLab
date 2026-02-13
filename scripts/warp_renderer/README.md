# Newton Warp Renderer in IsaacLab


### Install IsaacSim / IsaacLab
Install IsaacLab following this [guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)

```bash
git clone git@github.com:daniela-hase/IsaacLab.git
cd IsaacLab

uv venv --python 3.11 --seed
source .venv/bin/activate

uv pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

./isaaclab.sh --install
```

### Run the example
```bash
python ./scripts/warp_renderer/example.py --headless --steps 200 --num_envs 4 --enable_cameras
```

You can add `--save_images` to save the rendered images.


### How to use the Renderer

If you look at `example.py` you'll see these lines:
```python
tiled_camera_cfg: TiledCameraCfg = TiledCameraCfg(
    prim_path="/World/envs/env_.*/Camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(-3.0, 0.0, 1.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
    data_types=["rgb"],
    spawn=isaaclab_sim.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)),
    width=400,
    height=300,
    renderer="newton"
)
scene.sensors["tiled_camera"] = TiledCamera(tiled_camera_cfg, scene)
```

If the `renderer` parameter of `TiledCameraCfg` is `"newton"` the `NewtonWarpRenderer` will be used by the `TiledCamera`.

Please note: The `TiledCamera` constructor now takes an additional parameter `scene` to initialize the `NewtonWarpRenderer` with.
