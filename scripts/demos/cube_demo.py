# Falling Cube Demo
# Creates a single cube above the ground and lets it fall due to gravity.

from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.utils import prims as prim_utils

# Launch Isaac Sim app
app_launcher = AppLauncher(headless=False)  # False = show simulation window
simulation_app = app_launcher.app

# Import after simulation starts
from omni.isaac.lab.simulation_context import SimulationContext

# Create simulation context
sim = SimulationContext()
sim.reset()

# Ground plane
prim_utils.create_prim(
    "/World/GroundPlane",
    "Plane",
    translation=(0.0, 0.0, 0.0),
    scale=(10.0, 10.0, 1.0),
    color=(0.6, 0.6, 0.6)
)

# Cube above the ground (falls with gravity)
prim_utils.create_prim(
    "/World/FallingCube",
    "Cube",
    translation=(0.0, 0.0, 3.0),  # 3 meters above ground
    scale=(0.5, 0.5, 0.5),
    color=(1.0, 0.0, 0.0)  # Red cube
)

print("Running Falling Cube Demo... Press Ctrl+C to exit.")

# Step simulation until window is closed
while simulation_app.is_running():
    sim.step()

# Close app
simulation_app.close()
