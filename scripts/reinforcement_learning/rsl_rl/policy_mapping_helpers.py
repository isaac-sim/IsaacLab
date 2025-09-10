# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy mapping utilities for schema joint order transformations."""

import copy
import numpy as np
import torch
import os
import yaml
import tempfile
from typing import cast

import omni.usd
import isaaclab.sim as sim_utils

try:
    from tensordict import TensorDictBase as _TensorDictBase  # type: ignore
    TensorDictBase = _TensorDictBase
except Exception:
    TensorDictBase = tuple()  # fallback: isinstance(obs, TensorDictBase) will be False


class SchemaJointOrderHelperBase:
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

    def _get_term_type(self, term_func):
        """Get the observation type of a term function."""
        # Check if function has the generic_io_descriptor with observation_type
        if hasattr(term_func, '__wrapped__'):
            # For decorated functions, check for observation_type attribute
            if hasattr(term_func, 'observation_type'):
                return getattr(term_func, 'observation_type')
        
        # Check function name patterns as ultimate fallback
        func_name = getattr(term_func, '__name__', str(term_func))
        if 'joint_pos' in func_name or 'joint_vel' in func_name:
            return 'JointState'
        elif 'action' in func_name:
            return 'Action'
            
        return None

    def _validate_joint_terms(self, joint_names: list[str], term_names: list[str], term_dims: list[int]):
        """Find all joint-related observation terms (JointState and Action types)."""
        if not hasattr(self.base_env, "observation_manager"):
            return None
            
        obs_mgr = self.base_env.observation_manager
        if "policy" not in obs_mgr._group_obs_term_cfgs:
            return None
            
        term_cfgs = obs_mgr._group_obs_term_cfgs["policy"]
        
        # Find all joint-related terms (JointState and Action types)
        joint_related_indices = []
        
        for i, (term_name, term_cfg) in enumerate(zip(term_names, term_cfgs)):
            if not hasattr(term_cfg, 'func'):
                continue
                
            term_type = self._get_term_type(term_cfg.func)
            
            # All joint-related terms use the same permutation
            if term_type in ['JointState', 'Action']:
                # Validate size matches number of joints
                num_joints = len(joint_names)
                if term_dims[i] == num_joints:
                    joint_related_indices.append(i)
        
        # Need at least one joint-related term
        if not joint_related_indices:
            return None
        
        return joint_related_indices

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

    def _compute_observation_permutation(self, mappings, term_names: list[str], term_dims: list[int], joint_related_indices: list[int], for_import: bool):
        """Compute observation permutation. Direction controlled by for_import parameter."""
        engine_to_schema, schema_to_engine = mappings

        # Build flat observation permutation
        offsets = np.cumsum([0] + term_dims[:-1]).tolist()
        total_obs = int(np.sum(term_dims))
        obs_perm = np.arange(total_obs)

        # Apply same permutation to all joint-related terms
        for term_index in joint_related_indices:
            start = offsets[term_index]
            length = term_dims[term_index]
            
            if for_import:
                # For importing: build schema-ordered obs by selecting engine indices per schema index
                perm_slice = np.array(schema_to_engine, dtype=np.int64)
            else:
                # For exporting: build sim-ordered obs using inverse mapping (engine index -> schema index)  
                perm_slice = np.array(engine_to_schema, dtype=np.int64)
            
            obs_perm[start : start + length] = start + perm_slice

        return torch.as_tensor(obs_perm, dtype=torch.long), engine_to_schema, schema_to_engine

class SchemaImportHelper(SchemaJointOrderHelperBase):
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
        joint_related_indices = self._validate_joint_terms(engine_joint_names, term_names, term_dims)
        if joint_related_indices is None:
            return False

        # Compute observation permutation (for import, for_import=True)
        self.obs_perm, engine_to_schema, schema_to_engine = self._compute_observation_permutation(
            mappings, term_names, term_dims, joint_related_indices, for_import=True
        )
        # For import: action_perm maps schema actions -> engine actions
        self.action_perm = torch.as_tensor(engine_to_schema, dtype=torch.long)
        
        # print("obs_perm", self.obs_perm)
        # print("action_perm", self.action_perm)
        return True
        
class SchemaExportHelper(SchemaJointOrderHelperBase):
    """Export helper: compute observation/action reordering from Engine to Schema joint orders."""

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
        
        # Get observation terms info
        term_names, term_dims = self._get_observation_terms_info()
        if term_names is None or term_dims is None:
            return False

        # Validate joint terms
        joint_related_indices = self._validate_joint_terms(sim_joint_names, term_names, term_dims)
        if joint_related_indices is None:
            return False

        # Compute observation permutation (for export, for_import=False)
        self.obs_perm, engine_to_schema, schema_to_engine = self._compute_observation_permutation(
            mappings, term_names, term_dims, joint_related_indices, for_import=False
        )
        # For export: action_out_indices maps sim actions -> schema order
        # Original used: schema_to_sim, which is equivalent to our schema_to_engine
        self.action_out_indices = torch.as_tensor(schema_to_engine, dtype=torch.long)
        
        # print("obs_perm", self.obs_perm)
        # print("action_out_indices", self.action_out_indices)
        return True


class SchemaOrderedTorchPolicyExporter(torch.nn.Module):
    """Exporter that wraps policy to accept schema-ordered obs and emit schema-ordered actions."""

    def __init__(self, policy, normalizer, perm_helper: SchemaExportHelper):
        super().__init__()
        if getattr(policy, "is_recurrent", False):
            raise NotImplementedError("Schema-ordered export supports only non-recurrent policies.")
        # Ensure permutations are available for type-checkers and runtime
        assert perm_helper.obs_perm is not None
        assert perm_helper.action_out_indices is not None
        # deep copy actor/student
        if hasattr(policy, "actor"):
            self.actor = (
                torch.nn.Sequential(*[m for m in policy.actor.children()])
                if isinstance(policy.actor, torch.nn.Sequential)
                else copy.deepcopy(policy.actor)
            )
        elif hasattr(policy, "student"):
            self.actor = (
                torch.nn.Sequential(*[m for m in policy.student.children()])
                if isinstance(policy.student, torch.nn.Sequential)
                else copy.deepcopy(policy.student)
            )
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # copy normalizer
        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()
        # store permutations
        self.register_buffer("obs_perm", perm_helper.obs_perm.clone())
        self.register_buffer("action_out_indices", perm_helper.action_out_indices.clone())

    def _apply_obs_perm(self, x: torch.Tensor) -> torch.Tensor:
        # print("applying mapping from schema to sim with obs_perm", self.obs_perm)
        return x.index_select(dim=1, index=self.obs_perm)

    def _apply_action_perm(self, actions_sim: torch.Tensor) -> torch.Tensor:
        # print("applying mapping from sim to schema with action_out_indices", self.action_out_indices)
        return actions_sim.index_select(dim=1, index=self.action_out_indices)

    def forward(self, x):
        x = self._apply_obs_perm(x)
        actions_sim = self.actor(self.normalizer(x))
        return self._apply_action_perm(actions_sim)

    @torch.jit.export
    def reset(self):
        pass


def export_robot_schema_policy(
    base_env,
    runner,
    policy_nn,
    normalizer,
    export_model_dir: str,
    robot_schema_file: str | None,
):
    """Export schema-ordered policy artifacts.

    Exports:
    - JIT wrapper that accepts schema-ordered observations and emits schema-ordered actions (policy_schema_order.pt)
    - Runner checkpoint with weights remapped to schema order (policy_runner_schema_order.pt)
    """
    try:
        schema_override = None
        if robot_schema_file:
            with open(robot_schema_file) as f:
                cfg_yaml = yaml.safe_load(f)
            key = "robot_schema_joint_names"
            if key not in cfg_yaml:
                raise KeyError(f"Key '{key}' not found in YAML {robot_schema_file}")
            schema_override = list(cfg_yaml[key])

        perm_helper = SchemaExportHelper(base_env, policy_nn, normalizer, schema_override_names=schema_override)
        if perm_helper.compute():
            # Export schema-ordered JIT policy
            schema_jit = SchemaOrderedTorchPolicyExporter(policy_nn, normalizer, perm_helper)
            schema_jit.to("cpu")
            traced = torch.jit.script(schema_jit)
            schema_jit_path = os.path.join(export_model_dir, "policy_schema_order.pt")
            traced.save(schema_jit_path)
            print("[INFO] Exported schema-ordered JIT policy to:", schema_jit_path)

            # Additionally export a runner-compatible checkpoint for convenience
            try:
                runner_ckpt_path = os.path.join(export_model_dir, "policy_runner_schema_order.pt")

                # First save the original runner to get the proper checkpoint format
                # Use temporary directory to avoid side-effects in export directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_runner_path = os.path.join(temp_dir, "temp_runner.pt")

                    # Temporarily set up logging attributes for the original save
                    orig_log_dir = getattr(runner, 'log_dir', None)
                    orig_logger_type = getattr(runner, 'logger_type', None)
                    try:
                        if not hasattr(runner, 'logger_type'):
                            runner.logger_type = "tensorboard"
                        if getattr(runner, 'log_dir', None) is None:
                            runner.log_dir = temp_dir

                        # Save and load checkpoint to obtain proper serialization format
                        runner.save(temp_runner_path)
                        checkpoint = torch.load(temp_runner_path, map_location='cpu')
                    finally:
                        if orig_log_dir is not None:
                            runner.log_dir = orig_log_dir
                        elif hasattr(runner, 'log_dir'):
                            runner.log_dir = None
                        if orig_logger_type is not None:
                            runner.logger_type = orig_logger_type
                        elif hasattr(runner, 'logger_type') and orig_logger_type is None:
                            try:
                                delattr(runner, 'logger_type')
                            except AttributeError:
                                pass

                # Apply schema mapping to the checkpoint weights
                schema_checkpoint = copy.deepcopy(checkpoint)
                temp_policy = copy.deepcopy(policy_nn)

                obs_perm = perm_helper.obs_perm
                action_out_indices = perm_helper.action_out_indices
                assert obs_perm is not None
                assert action_out_indices is not None
                inv_obs_perm = torch.empty_like(obs_perm)
                inv_obs_perm[obs_perm] = torch.arange(
                    obs_perm.numel(), device=obs_perm.device, dtype=obs_perm.dtype
                )

                # Reorder normalizer if present
                if hasattr(temp_policy, "actor_obs_normalizer"):
                    norm = temp_policy.actor_obs_normalizer
                    if norm is not None:
                        try:
                            sd = norm.state_dict()
                            for k, v in list(sd.items()):
                                if (
                                    isinstance(v, torch.Tensor)
                                    and v.dim() == 1
                                    and v.numel() == inv_obs_perm.numel()
                                ):
                                    sd[k] = v.index_select(0, inv_obs_perm.to(v.device))
                            norm.load_state_dict(sd, strict=False)
                        except Exception:
                            pass

                # Reorder first and last linear layers in actor/student
                actor_module = getattr(temp_policy, "actor", None) or getattr(temp_policy, "student", None)
                if actor_module is not None:
                    with torch.no_grad():
                        first_linear = None
                        last_linear = None
                        for m in actor_module.modules():
                            if isinstance(m, torch.nn.Linear):
                                if first_linear is None:
                                    first_linear = m
                                last_linear = m

                        if first_linear is not None:
                            idx = inv_obs_perm.to(first_linear.weight.device)
                            first_linear.weight.data = first_linear.weight.data.index_select(1, idx)

                        if last_linear is not None:
                            aidx = cast(torch.Tensor, action_out_indices).to(last_linear.weight.device)
                            last_linear.weight.data = last_linear.weight.data.index_select(0, aidx)
                            if last_linear.bias is not None:
                                last_linear.bias.data = last_linear.bias.data.index_select(0, aidx)

                schema_checkpoint["model_state_dict"] = temp_policy.state_dict()
                torch.save(schema_checkpoint, runner_ckpt_path)
                print("[INFO] Exported schema-ordered runner checkpoint to:", runner_ckpt_path)
            except Exception as e:
                print(f"[WARN] Failed to export schema-ordered runner checkpoint: {e}")
        else:
            print("[WARN] Could not compute schema joint order mapping; skipping schema-ordered exports.")
    except Exception as e:
        print(f"[WARN] Schema-ordered export failed: {e}")


def import_robot_schema_policy(
    base_env,
    robot_schema_file: str | None,
):
    """Return observation and action remap callables for schema import.

    Returns a tuple: (obs_remap_fn, action_remap_fn). On failure, returns (None, None).
    """
    obs_remap_fn = None
    action_remap_fn = None
    try:
        schema_override = None
        if robot_schema_file:
            with open(robot_schema_file) as f:
                cfg_yaml = yaml.safe_load(f)
            key = "robot_schema_joint_names"
            if key not in cfg_yaml:
                raise KeyError(f"Key '{key}' not found in YAML {robot_schema_file}")
            schema_override = list(cfg_yaml[key])

        import_helper = SchemaImportHelper(base_env, schema_override_names=schema_override)
        if import_helper.compute():
            print("[INFO] Successfully computed schema import mappings")
            # Stabilize types for static analysis
            obs_perm_t = cast(torch.Tensor, import_helper.obs_perm)
            action_perm_t = cast(torch.Tensor, import_helper.action_perm)

            def _obs_remap_fn(obs):
                # TensorDict support
                if isinstance(obs, TensorDictBase):
                    if "policy" in obs.keys():
                        obs_copy = obs.clone()
                        obs_copy["policy"] = obs_copy["policy"].index_select(
                            dim=1, index=obs_perm_t.to(obs_copy["policy"].device)
                        )
                        return obs_copy
                    else:
                        print("[WARN] TensorDict missing 'policy' key; skipping remap")
                        return obs
                # dict-like
                if isinstance(obs, dict):
                    if "policy" in obs:
                        obs_copy = obs.copy()
                        obs_copy["policy"] = obs_copy["policy"].index_select(
                            dim=1, index=obs_perm_t.to(obs_copy["policy"].device)
                        )
                        return obs_copy
                    else:
                        print("[WARN] Dict missing 'policy' key; skipping remap")
                        return obs
                # tensor
                if hasattr(obs, "index_select"):
                    return obs.index_select(dim=1, index=obs_perm_t.to(obs.device))

                print(f"[WARN] Unsupported observation type for remapping: {type(obs)}")
                return obs

            def _action_remap_fn(actions):
                return actions.index_select(dim=1, index=action_perm_t.to(actions.device))

            obs_remap_fn = _obs_remap_fn
            action_remap_fn = _action_remap_fn
            print("[INFO] Schema import remapping functions enabled")
        else:
            print("[WARN] Could not compute schema joint order mapping for import; using original policy without remapping.")
    except Exception as e:
        print(f"[WARN] Schema import failed: {e}")
    return obs_remap_fn, action_remap_fn
