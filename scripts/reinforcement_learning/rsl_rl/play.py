# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
from scripts.reinforcement_learning.rsl_rl import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--newton_visualizer", action="store_true", default=False, help="Enable Newton rendering.")

parser.add_argument(
    "--import_schema_joint_order",
    action="store_true", 
    default=False,
    help="Import policy from schema joint order representation to current engine representation.",
)
parser.add_argument(
    "--export_schema_joint_order",
    action="store_true",
    default=False,
    help="Export additional JIT/ONNX policies using USD Isaac Robot Schema joint order.",
)
parser.add_argument(
    "--schema_joint_order_file",
    type=str,
    default=None,
    help="Path to YAML file containing joint order to treat as schema order for import/export (uses key 'robot_schema_joint_names' by default).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
import copy
import yaml

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.timer import Timer

Timer.enable = False
Timer.enable_display_output = False

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


# Helper imports for USD Isaac Robot Schema lookup and prim resolution
from pxr import Usd
import omni.usd
import isaaclab.sim as sim_utils
try:
    from tensordict import TensorDictBase as _TensorDictBase  # type: ignore
    TensorDictBase = _TensorDictBase
except Exception:
    TensorDictBase = tuple()  # fallback: isinstance(obs, TensorDictBase) will be False


class _SchemaJointOrderHelperBase:
    """Base class for schema joint order helpers with common functionality."""

    def __init__(self, base_env, schema_override_names: list[str] | None = None):
        self.base_env = base_env
        self._schema_override_names = list(schema_override_names) if schema_override_names else None

    def _get_scene_articulation_and_joint_names(self):
        """Get articulation and joint names from the current environment."""
        scene = self.base_env.scene
        # Prefer common key 'robot', else fallback to the first articulation
        if hasattr(scene, "articulations") and isinstance(scene.articulations, dict) and len(scene.articulations) > 0:
            if "robot" in scene.articulations:
                art = scene.articulations["robot"]
            else:
                art = next(iter(scene.articulations.values()))
            return art, list(art.joint_names)
        return None, None

    def _get_schema_joint_names(self, art) -> list[str] | None:
        """Get joint names from USD Isaac Robot Schema."""
        try:
            # Resolve the robot prim in the first environment
            first_robot_prim = sim_utils.find_first_matching_prim(art.cfg.prim_path)
            if first_robot_prim is None:
                return None
            stage = omni.usd.get_context().get_stage()
            prim = first_robot_prim
            # Import here to avoid hard dependency if schema package is unavailable
            from usd.schema.isaac import robot_schema  # type: ignore

            joints = robot_schema.utils.GetAllRobotJoints(stage, prim, False)
            schema_joint_names = []
            for j in joints:
                # joints may be prims or have GetPrim(); robustly extract name
                try:
                    p = j.GetPrim() if hasattr(j, "GetPrim") else j
                    name = p.GetPath().pathString.rsplit("/", 1)[-1]
                except Exception:
                    name = str(j)
                schema_joint_names.append(name)
            return schema_joint_names
        except Exception:
            return None

    def _get_observation_terms_info(self):
        """Extract observation manager information."""
        if hasattr(self.base_env, "observation_manager"):
            obs_mgr = self.base_env.observation_manager
            if ("policy" not in obs_mgr.active_terms) or ("policy" not in obs_mgr.group_obs_term_dim):
                return None, None
            term_names = list(obs_mgr.active_terms["policy"])  # list[str]
            term_dims = [int(np.prod(d)) for d in obs_mgr.group_obs_term_dim["policy"]]
            return term_names, term_dims
        return None, None

    def _validate_joint_terms(self, joint_names: list[str], term_names: list[str], term_dims: list[int]):
        """Validate joint-related observation terms."""
        try:
            idx_joint_pos = term_names.index("joint_pos")
            idx_joint_vel = term_names.index("joint_vel")
            idx_actions_obs = term_names.index("actions")
        except ValueError:
            return None

        # Validate sizes match number of joints
        num_joints = len(joint_names)
        if (
            term_dims[idx_joint_pos] != num_joints
            or term_dims[idx_joint_vel] != num_joints
            or term_dims[idx_actions_obs] != num_joints
        ):
            return None

        return idx_joint_pos, idx_joint_vel, idx_actions_obs

    def _build_joint_index_mappings(self, engine_joint_names: list[str], schema_joint_names: list[str]):
        """Build bidirectional mappings between engine and schema joint orders."""
        # Filter schema list to only those joints that exist in engine
        schema_filtered = [n for n in schema_joint_names if n in engine_joint_names]
        if len(schema_filtered) != len(engine_joint_names):
            return None, None
        
        engine_index = {n: i for i, n in enumerate(engine_joint_names)}
        schema_index = {n: i for i, n in enumerate(schema_filtered)}
        
        # engine_to_schema: for each engine joint index, what schema index should it map to
        engine_to_schema = [schema_index[n] for n in engine_joint_names]
        # schema_to_engine: for each schema joint index, what engine index should it map to  
        schema_to_engine = [engine_index[n] for n in schema_filtered]
        
        return engine_to_schema, schema_to_engine

    def _compute_observation_permutation(self, mappings, term_names: list[str], term_dims: list[int], term_indices, for_import: bool):
        """Compute observation permutation. Direction controlled by for_import parameter."""
        engine_to_schema, schema_to_engine = mappings
        idx_joint_pos, idx_joint_vel, idx_actions_obs = term_indices

        # Build flat observation permutation
        offsets = np.cumsum([0] + term_dims[:-1]).tolist()
        total_obs = int(np.sum(term_dims))
        obs_perm = np.arange(total_obs)

        # For each joint-related slice, set the appropriate permutation
        for term_index in (idx_joint_pos, idx_joint_vel, idx_actions_obs):
            start = offsets[term_index]
            length = term_dims[term_index]
            
            if for_import:
                # For importing: build schema-ordered obs by selecting engine indices per schema index
                # Original import used: schema_to_engine
                perm_slice = np.array(schema_to_engine, dtype=np.int64)
            else:
                # For exporting: build sim-ordered obs using inverse mapping (engine index -> schema index)  
                # Original export used: sim_to_schema (which maps sim_index -> schema_index)
                # This is equivalent to our engine_to_schema
                perm_slice = np.array(engine_to_schema, dtype=np.int64)
            
            obs_perm[start : start + length] = start + perm_slice

        return torch.as_tensor(obs_perm, dtype=torch.long), engine_to_schema, schema_to_engine


class _SchemaPermutationHelper(_SchemaJointOrderHelperBase):
    """Compute observation/action reordering between Schema and Simulator joint orders for export."""

    def __init__(self, base_env, policy_module, normalizer, schema_override_names: list[str] | None = None):
        super().__init__(base_env, schema_override_names)
        self.policy_module = policy_module
        self.normalizer = normalizer
        self.is_recurrent = getattr(policy_module, "is_recurrent", False)
        # filled by compute()
        self.obs_perm = None  # 1D LongTensor of size num_obs to map schema-ordered obs -> sim-ordered obs
        self.action_out_indices = None  # 1D LongTensor of size num_actions to map sim actions -> schema order


    def compute(self) -> bool:
        """Compute permutations for export."""
        # Get sim joint names
        art, sim_joint_names = self._get_scene_articulation_and_joint_names()
        if art is None or not sim_joint_names:
            return False
        
        # Get schema joint names
        schema_joint_names = self._schema_override_names if self._schema_override_names is not None else self._get_schema_joint_names(art)
        if not schema_joint_names:
            return False
        
        # Build index mappings
        mappings = self._build_joint_index_mappings(sim_joint_names, schema_joint_names)
        if mappings[0] is None:
            return False
        
        # print(mappings[0], mappings[1])
        
        # Get observation terms info
        term_names, term_dims = self._get_observation_terms_info()
        if term_names is None or term_dims is None:
            return False

        # Validate joint terms
        term_indices = self._validate_joint_terms(sim_joint_names, term_names, term_dims)
        if term_indices is None:
            return False

        # Compute observation permutation (for export, for_import=False)
        self.obs_perm, engine_to_schema, schema_to_engine = self._compute_observation_permutation(
            mappings, term_names, term_dims, term_indices, for_import=False
        )
        # For export: action_out_indices maps sim actions -> schema order
        # Original used: schema_to_sim, which is equivalent to our schema_to_engine
        self.action_out_indices = torch.as_tensor(schema_to_engine, dtype=torch.long)
        
        # print("obs_perm", self.obs_perm)
        # print("action_out_indices", self.action_out_indices)
        return True


class _SchemaOrderedTorchPolicyExporter(torch.nn.Module):
    """Exporter that wraps policy to accept schema-ordered obs and emit schema-ordered actions."""

    def __init__(self, policy, normalizer, perm_helper: _SchemaPermutationHelper):
        super().__init__()
        if getattr(policy, "is_recurrent", False):
            raise NotImplementedError("Schema-ordered export supports only non-recurrent policies.")
        # deep copy actor/student
        if hasattr(policy, "actor"):
            self.actor = torch.nn.Sequential(*[m for m in policy.actor.children()]) if isinstance(policy.actor, torch.nn.Sequential) else copy.deepcopy(policy.actor)
        elif hasattr(policy, "student"):
            self.actor = torch.nn.Sequential(*[m for m in policy.student.children()]) if isinstance(policy.student, torch.nn.Sequential) else copy.deepcopy(policy.student)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # copy normalizer
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()
        # store permutations
        self.register_buffer("obs_perm", perm_helper.obs_perm.clone())
        self.register_buffer("action_out_indices", perm_helper.action_out_indices.clone())

    def _apply_obs_perm(self, x: torch.Tensor) -> torch.Tensor:
        return x.index_select(dim=1, index=self.obs_perm)

    def _apply_action_perm(self, actions_sim: torch.Tensor) -> torch.Tensor:
        return actions_sim.index_select(dim=1, index=self.action_out_indices)

    def forward(self, x):
        x = self._apply_obs_perm(x)
        actions_sim = self.actor(self.normalizer(x))
        return self._apply_action_perm(actions_sim)

    @torch.jit.export
    def reset(self):
        pass


class _SchemaImportHelper(_SchemaJointOrderHelperBase):
    """Helper to import policies from schema joint order representation to engine representation."""
    
    def __init__(self, base_env, schema_override_names: list[str] | None = None):
        super().__init__(base_env, schema_override_names)
        # filled by compute()
        self.obs_perm = None  # 1D LongTensor to map engine obs -> schema obs (for input to policy)
        self.action_perm = None  # 1D LongTensor to map schema actions -> engine actions (for output from policy)
        
    
    def compute(self) -> bool:
        """Compute the permutation mappings for importing from schema representation."""
        # Get engine joint names
        art, engine_joint_names = self._get_scene_articulation_and_joint_names()
        if art is None or not engine_joint_names:
            return False
            
        # Get schema joint names
        schema_joint_names = self._schema_override_names if self._schema_override_names is not None else self._get_schema_joint_names(art)
        if not schema_joint_names:
            return False
            
        # Build index mappings
        mappings = self._build_joint_index_mappings(engine_joint_names, schema_joint_names)
        if mappings[0] is None:
            return False
        
        # print("engine_to_schema", mappings[0])
        # print("schema_to_engine", mappings[1])
        
        # Get observation terms info
        term_names, term_dims = self._get_observation_terms_info()
        if term_names is None or term_dims is None:
            return False

        # Validate joint terms
        term_indices = self._validate_joint_terms(engine_joint_names, term_names, term_dims)
        if term_indices is None:
            return False

        # Compute observation permutation (for import, for_import=True)
        self.obs_perm, engine_to_schema, schema_to_engine = self._compute_observation_permutation(
            mappings, term_names, term_dims, term_indices, for_import=True
        )
        # For import: action_perm maps schema actions -> engine actions
        self.action_perm = torch.as_tensor(engine_to_schema, dtype=torch.long)
        
        # print("obs_perm", self.obs_perm)
        # print("action_perm", self.action_perm)
        return True


def main():
    """Play with RSL-RL agent.
    You can use this script to export a policy in schema order, and import a policy from schema order to the current engine representation.
    To export a policy in schema order, you can use the following command:
    Example:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py\
    --task=Isaac-Velocity-Flat-Anymal-D-v0 \
    --num_envs=32 \
    --export_schema_joint_order \
    --schema_joint_order_file ../IsaacLab/scripts/newton_sim2sim/mappings/sim2sim_anymal_d.yaml

    This will save JIT and runner checkpoint in the exported directory. You can use this to import the policy to the physX-based Isaac Lab.
    To import a policy from schema order, you can use the following command:
    Example:
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py\
    --task=Isaac-Velocity-Flat-Anymal-D-v0 \
    --num_envs=32 \
    --import_schema_joint_order \
    --schema_joint_order_file ../IsaacLab/scripts/newton_sim2sim/mappings/sim2sim_anymal_d.yaml \
    --checkpoint /path/to/exported/policy_runner_schema_order.pt

    """
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        newton_visualizer=args_cli.newton_visualizer,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic
    # Move the policy used by the runner to the env device to avoid mixed devices
    try:
        ppo_runner.alg.to(env.unwrapped.device)  # type: ignore[attr-defined]
    except Exception:
        pass
    inference_device = env.unwrapped.device

    # export policy to onnx/jit (sim joint order)
    # Ensure we don't create nested "exported" directories
    checkpoint_dir = os.path.dirname(resume_path)
    if os.path.basename(checkpoint_dir) == "exported":
        # If resume_path is already in an exported directory, go up one level
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    export_model_dir = os.path.join(checkpoint_dir, "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # Optionally export schema-ordered policy variants
    if args_cli.export_schema_joint_order:
        try:
            schema_override = None
            if args_cli.schema_joint_order_file:
                with open(args_cli.schema_joint_order_file) as f:
                    cfg_yaml = yaml.safe_load(f)
                key = "robot_schema_joint_names"
                if key not in cfg_yaml:
                    raise KeyError(f"Key '{key}' not found in YAML {args_cli.schema_joint_order_file}")
                schema_override = list(cfg_yaml[key])

            perm_helper = _SchemaPermutationHelper(env.unwrapped, policy_nn, ppo_runner.obs_normalizer, schema_override_names=schema_override)
            if perm_helper.compute():
                # Export schema-ordered JIT policy with _schema_order suffix
                schema_jit = _SchemaOrderedTorchPolicyExporter(policy_nn, ppo_runner.obs_normalizer, perm_helper)
                schema_jit.to("cpu")
                traced = torch.jit.script(schema_jit)
                schema_jit_path = os.path.join(export_model_dir, "policy_schema_order.pt")
                traced.save(schema_jit_path)
                print("[INFO] Exported schema-ordered JIT policy to:", schema_jit_path)
                # Additionally export a runner-compatible checkpoint for convenience
                try:
                    runner_ckpt_path = os.path.join(export_model_dir, "policy_runner_schema_order.pt")
                    
                    # First save the original runner to get the proper checkpoint format
                    import tempfile
                    
                    # Create a temporary directory for the runner save operation
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_runner_path = os.path.join(temp_dir, "temp_runner.pt")
                        
                        # Temporarily set up logging attributes for the original save
                        orig_log_dir = getattr(ppo_runner, 'log_dir', None)
                        orig_logger_type = getattr(ppo_runner, 'logger_type', None)
                        
                        try:
                            # Set up minimal logging attributes for save operation
                            # Use the temporary directory to avoid creating subdirs in export_model_dir
                            if not hasattr(ppo_runner, 'logger_type'):
                                ppo_runner.logger_type = "tensorboard"
                            if getattr(ppo_runner, 'log_dir', None) is None:
                                ppo_runner.log_dir = temp_dir
                            
                            # Save original runner to temp file
                            ppo_runner.save(temp_runner_path)
                            
                            # Load the checkpoint to get the proper format
                            checkpoint = torch.load(temp_runner_path, map_location='cpu')
                            
                        finally:
                            # Restore original logging attributes
                            if orig_log_dir is not None:
                                ppo_runner.log_dir = orig_log_dir
                            elif hasattr(ppo_runner, 'log_dir'):
                                ppo_runner.log_dir = None
                                
                            if orig_logger_type is not None:
                                ppo_runner.logger_type = orig_logger_type
                            elif hasattr(ppo_runner, 'logger_type') and orig_logger_type is None:
                                try:
                                    delattr(ppo_runner, 'logger_type')
                                except AttributeError:
                                    pass
                        
                        # temp_dir and temp_runner_path are automatically cleaned up
                    
                    # Apply schema mapping to the checkpoint
                    schema_checkpoint = copy.deepcopy(checkpoint)
                    
                    # Create temporary policy to apply mapping
                    temp_policy = copy.deepcopy(policy_nn)
                    
                    # Apply schema reordering
                    obs_perm = perm_helper.obs_perm
                    action_out_indices = perm_helper.action_out_indices
                    inv_obs_perm = torch.empty_like(obs_perm)
                    inv_obs_perm[obs_perm] = torch.arange(obs_perm.numel(), device=obs_perm.device, dtype=obs_perm.dtype)
                    
                    # Apply reordering to temp policy
                    if hasattr(temp_policy, "actor_obs_normalizer"):
                        norm = temp_policy.actor_obs_normalizer
                        if norm is not None:
                            try:
                                sd = norm.state_dict()
                                for k, v in list(sd.items()):
                                    if isinstance(v, torch.Tensor) and v.dim() == 1 and v.numel() == inv_obs_perm.numel():
                                        sd[k] = v.index_select(0, inv_obs_perm.to(v.device))
                                norm.load_state_dict(sd, strict=False)
                            except Exception:
                                pass
                    
                    # Apply to actor/student
                    actor_module = getattr(temp_policy, "actor", None) or getattr(temp_policy, "student", None)
                    if actor_module is not None:
                        with torch.no_grad():
                            # Find first and last linear layers
                            first_linear = None
                            last_linear = None
                            for m in actor_module.modules():
                                if isinstance(m, torch.nn.Linear):
                                    if first_linear is None:
                                        first_linear = m
                                    last_linear = m
                            
                            # Reorder first linear input
                            if first_linear is not None:
                                idx = inv_obs_perm.to(first_linear.weight.device)
                                first_linear.weight.data = first_linear.weight.data.index_select(1, idx)
                            
                            # Reorder last linear output
                            if last_linear is not None:
                                aidx = action_out_indices.to(last_linear.weight.device)
                                last_linear.weight.data = last_linear.weight.data.index_select(0, aidx)
                                if last_linear.bias is not None:
                                    last_linear.bias.data = last_linear.bias.data.index_select(0, aidx)
                    
                    # Update checkpoint with modified state
                    schema_checkpoint['model_state_dict'] = temp_policy.state_dict()
                    
                    # Save the schema checkpoint
                    torch.save(schema_checkpoint, runner_ckpt_path)
                    print("[INFO] Exported schema-ordered runner checkpoint to:", runner_ckpt_path)
                except Exception as e:
                    print(f"[WARN] Failed to export schema-ordered runner checkpoint: {e}")
            else:
                print("[WARN] Could not compute schema joint order mapping; skipping schema-ordered exports.")
        except Exception as e:
            print(f"[WARN] Schema-ordered export failed: {e}")

    # Schema import functionality - remap observations and actions for imported policies
    obs_remap_fn = None
    action_remap_fn = None
    if args_cli.import_schema_joint_order:
        try:
            schema_override = None
            if args_cli.schema_joint_order_file:
                with open(args_cli.schema_joint_order_file) as f:
                    cfg_yaml = yaml.safe_load(f)
                key = "robot_schema_joint_names"
                if key not in cfg_yaml:
                    raise KeyError(f"Key '{key}' not found in YAML {args_cli.schema_joint_order_file}")
                schema_override = list(cfg_yaml[key])

            import_helper = _SchemaImportHelper(env.unwrapped, schema_override_names=schema_override)
            if import_helper.compute():
                print("[INFO] Successfully computed schema import mappings")
                
                def obs_remap_fn(obs):
                    """Remap engine observations to schema order for policy input."""
                    # TensorDict support
                    if isinstance(obs, TensorDictBase):
                        if "policy" in obs.keys():
                            obs_copy = obs.clone()
                            obs_copy["policy"] = obs_copy["policy"].index_select(dim=1, index=import_helper.obs_perm.to(obs_copy["policy"].device))
                            return obs_copy
                        else:
                            print("[WARN] TensorDict missing 'policy' key; skipping remap")
                            return obs
                    # dict-like (plain dict)
                    if isinstance(obs, dict):
                        if "policy" in obs:
                            obs_copy = obs.copy()
                            obs_copy["policy"] = obs_copy["policy"].index_select(dim=1, index=import_helper.obs_perm.to(obs_copy["policy"].device))
                            return obs_copy
                        else:
                            print("[WARN] Dict missing 'policy' key; skipping remap")
                            return obs
                    # tensor
                    if hasattr(obs, 'index_select'):
                        return obs.index_select(dim=1, index=import_helper.obs_perm.to(obs.device))
                    
                    print(f"[WARN] Unsupported observation type for remapping: {type(obs)}")
                    return obs
                
                def action_remap_fn(actions):
                    """Remap schema actions to engine order for environment stepping."""
                    return actions.index_select(dim=1, index=import_helper.action_perm.to(actions.device))
                    
                print("[INFO] Schema import remapping functions enabled")
            else:
                print("[WARN] Could not compute schema joint order mapping for import; using original policy without remapping.")
        except Exception as e:
            print(f"[WARN] Schema import failed: {e}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    # Align runner/policy devices with observation device
    try:
        if isinstance(obs, dict):
            if "policy" in obs and isinstance(obs["policy"], torch.Tensor):
                target_device = obs["policy"].device
            else:
                # pick first tensor value's device if available, else fallback to env device
                tensor_values = [v for v in obs.values() if isinstance(v, torch.Tensor)]
                target_device = tensor_values[0].device if tensor_values else env.unwrapped.device
        elif isinstance(obs, torch.Tensor):
            target_device = obs.device
        else:
            target_device = env.unwrapped.device

        if hasattr(ppo_runner, "alg") and hasattr(ppo_runner.alg, "to"):
            ppo_runner.alg.to(target_device)
        # Move runner's obs normalizer if present
        try:
            obs_norm = getattr(ppo_runner, "obs_normalizer", None)
            if obs_norm is not None and hasattr(obs_norm, "to"):
                obs_norm.to(target_device)
        except Exception:
            pass
        if hasattr(policy_nn, "to"):
            policy_nn.to(target_device)
        if hasattr(policy_nn, "actor") and isinstance(policy_nn.actor, torch.nn.Module):
            policy_nn.actor.to(target_device)
        if hasattr(policy_nn, "student") and isinstance(policy_nn.student, torch.nn.Module):
            policy_nn.student.to(target_device)
        if hasattr(policy_nn, "memory_a") and hasattr(policy_nn.memory_a, "rnn"):
            policy_nn.memory_a.rnn.to(target_device)
        if hasattr(policy_nn, "memory_s") and hasattr(policy_nn.memory_s, "rnn"):
            policy_nn.memory_s.rnn.to(target_device)
        
        # Also move the policy function itself
        if hasattr(policy, "to"):
            policy.to(target_device)
        

        
        # Derive final inference device from policy params to avoid stale state
        try:
            first_param = next(policy_nn.parameters())
            inference_device = first_param.device
        except Exception:
            inference_device = target_device
    except Exception:
        pass
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # move observations to policy device to avoid CPU/GPU mismatch
            obs = obs.to(inference_device)
            
            # Apply observation remapping if schema import is enabled
            policy_input = obs_remap_fn(obs) if obs_remap_fn else obs
            actions = policy(policy_input)
            
            # Apply action remapping if schema import is enabled
            env_actions = action_remap_fn(actions) if action_remap_fn else actions
            
            # ensure actions are on environment device for stepping
            env_actions = env_actions.to(env.unwrapped.device)
            # env stepping
            obs, _, _, _ = env.step(env_actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
