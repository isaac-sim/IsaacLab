# Isaac Lab: Assets for Robots and Objects

This extension contains configurations for various assets and sensors. The configuration instances are
used to spawn and configure the instances in the simulation. They are passed to their corresponding
classes during construction.

## Organizing custom assets

For Isaac Lab, we primarily store assets on the Omniverse Nucleus server. However, at times, it may be
needed to store the assets locally (for debugging purposes). In such cases, the extension's `data`
directory can be used for temporary hosting of assets.

Inside the `data` directory, we recommend following the same structure as our Nucleus directory
`Isaac/IsaacLab`. This helps us later to move these assets to the Nucleus server seamlessly.

The recommended directory structure inside `data` is as follows:

* **`Robots/<Company-Name>/<Robot-Name>`**: The USD files should be inside `<Robot-Name>` directory with
  the name of the robot.
* **`Props/<Prop-Type>/<Prop-Name>`**: The USD files should be inside `<Prop-Name>` directory with the name
  of the prop. This includes mounts, objects and markers.
* **`ActuatorNets/<Company-Name>`**: The actuator networks should inside `<Company-Name` directory with the
  name of the actuator that it models.
* **`Policies/<Task-Name>`**: The policy should be JIT/ONNX compiled with the name `policy.pt`. It should also
  contain the parameters used for training the checkpoint. This is to ensure reproducibility.
* **`Test/<Test-Name>`**: The asset used for unit testing purposes.

## Referring to the assets in your code

You can use the following snippet to refer to the assets:

```python

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR


# ANYmal-C
ANYMAL_C_USD_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
# ANYmal-D
ANYMAL_D_USD_PATH = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd"
```
