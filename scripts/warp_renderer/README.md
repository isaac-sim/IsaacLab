# Newton Warp Renderer in IsaacLab


### Install IsaacSim / IsaacLab
Install IsaacLab following this [guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)

```bash
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab

uv venv --python 3.11 --seed
source .venv/bin/activate

uv pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

./isaaclab.sh --install
```

### Newton Warp Renderer dependency
We need to provide the [Newton Warp Renderer](https://github.com/newton-physics/newton/tree/main/newton/_src/sensors/warp_raytrace) for this example to run.

In the long term this should probably be done by adding Newton as a PIP depenndency to IsaacLab or similar.

But for now, I just softlinked the `warp_raytrace` folder from the Newton repository over to here - so if you check out this branch and want to run the example, you need to do that as well :)

Just clone the [Newton](https://github.com/newton-physics/newton) repository and copy/paste or link the [warp_raytrace](https://github.com/newton-physics/newton/tree/main/newton/_src/sensors/warp_raytrace) folder here.


### Run the example
```bash
python ./scripts/warp_renderer/example.py --headless --steps 200 --num_envs 4 --enable_cameras --kit_args "--enable omni.warp.core-1.11.0-rc.1+lx64" 
```

The included Warp version with IsaacLab is too old for the Newton Warp Renderer, therefore we need to specify a newer version with the `--kit_args` parameter.

You can add `--save_images` to save the rendered images.


### How to use the Renderer

If you look at `example.py` you'll see these lines:
```python
renderer = WarpRenderer(scene, 400, 400)

# ...

def run_simulator(...):
    # ...
    renderer.update()
    renderer.render()
    if save_images:
        renderer.save_image(f"warp_renderer/rgb.{step:04d}.png")
```
Once the renderer is created, the `update()` method will write all transforms from the IsaacLab Fabric USD Stage to the Newton Warp Renderer buffers, which are then rendered by the `render()` method.