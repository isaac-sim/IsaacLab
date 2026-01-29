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
from isaaclab.renderers import NewtonWarpRenderer

renderer = NewtonWarpRenderer(scene, width, height)

# ...

def run_simulator(...):
    # ...
    renderer.update()
    renderer.render()
    if save_images:
        renderer.save_image(f"warp_renderer/rgb.{step:04d}.png")
```
Once the renderer is created, the `update()` method will write all transforms from the IsaacLab Fabric USD Stage to the Newton Warp Renderer buffers, which are then rendered by the `render()` method.