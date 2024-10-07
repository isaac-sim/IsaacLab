
from omni.isaac.lab.app import AppLauncher
# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app
import asyncio
import numpy as np
from omni.isaac.core.world import World
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.objects import DynamicCuboid

async def example():
    if World.instance():
        World.instance().clear_instance()
    world = World()
    world.scene.add_default_ground_plane()
    await world.initialize_simulation_context_async()

    # create three rigid cubes sitting on top of three others
    for i in range(3):
        DynamicCuboid(prim_path=f"/World/bottom_box_{i+1}", size=2, color=np.array([0.5, 0, 0]), mass=1.0)
        DynamicCuboid(prim_path=f"/World/top_box_{i+1}", size=2, color=np.array([0, 0, 0.5]), mass=1.0)

    # as before, create RigidContactView to manipulate bottom boxes but this time specify top boxes as filters to the view object
    # this allows receiving contact forces between the bottom boxes and top boxes
    bottom_box_view = RigidPrimView(
        prim_paths_expr="/World/bottom_box_*",
        positions=np.array([[0, 0, 1.0], [-5.0, 0, 1.0], [5.0, 0, 1.0]]),
        contact_filter_prim_paths_expr=["/World/top_box_*"],
    )
    # create a RigidContactView to manipulate top boxes
    top_box_view = RigidPrimView(
        prim_paths_expr="/World/top_box_*",
        positions=np.array([[0.0, 0, 3.0], [-5.0, 0, 3.0], [5.0, 0, 3.0]]),
        track_contact_forces=True,
    )

    world.scene.add(top_box_view)
    world.scene.add(bottom_box_view)
    await world.reset_async()

    time.sleep(15)
    # net contact forces acting on the bottom boxes
    print(bottom_box_view.get_net_contact_forces())
    # contact forces between the top and the bottom boxes
    print(bottom_box_view.get_contact_force_matrix())
    time.sleep(15)
asyncio.ensure_future(example())