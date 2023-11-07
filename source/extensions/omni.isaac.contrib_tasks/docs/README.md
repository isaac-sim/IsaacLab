# Orbit: Contributed Environment

This extension serves as a platform to host contributed environments from the robotics and machine learning
community. The extension follows the same style as the `omni.isaac.orbit_tasks` extension.

The environments should follow the `gym.Env` API from OpenAI Gym version `0.21.0`. They need to be registered using
the Gym registry.

To follow the same convention, each environment's name is composed of `Isaac-Contrib-<Task>-<Robot>-v<X>`,
where `<Task>` indicates the skill to learn in the environment, `<Robot>` indicates the embodiment of the
acting agent, and `<X>` represents the version of the environment (which can be used to suggest different
observation or action spaces).

The environments can be configured using either Python classes (wrapped using `configclass` decorator) or through
YAML files. The template structure of the environment is always put at the same level as the environment file
itself. However, its various instances should be included in directories within the environment directory itself.

The environments should then be registered in the `omni/isaac/contrib_tasks/__init__.py`:

```python
import gymnasium as gym

gym.register(
    id="Isaac-Contrib-<my-awesome-env>-v0",
    entry_point="omni.isaac.contrib_tasks.<your-env-package>:<your-env-class>",
    disable_env_checker=True,
    kwargs={"cfg_entry_point": "omni.isaac.contrib_tasks.<your-env-package-cfg>:<your-env-class-cfg>"},
)
```
