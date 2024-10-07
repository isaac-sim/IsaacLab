import numpy as np
import open3d
import torch
from einops import repeat

# from curobo.types.math import Pose
# from curobo.types.base import TensorDeviceType
# from curobo.types.robot import RobotConfig
# from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
# from curobo.wrap.reacher.ik_solver import IKSolverConfig, IKSolver
# from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
# from curobo.geom.types import WorldConfig, Cuboid, Mesh
# from curobo.util_file import get_assets_path, join_path, load_yaml

from pytorch3d.structures import Meshes
import pytorch_kinematics as pk


def set_connect_constraint(model, data, cid, pos):
    id1 = model.eq_obj1id[cid]
    id2 = model.eq_obj2id[cid]
    anchor1 = np.linalg.inv(data.xmat[id1].reshape(3, 3)) @ (pos - data.xpos[id1])
    anchor2 = np.linalg.inv(data.xmat[id2].reshape(3, 3)) @ (pos - data.xpos[id2])
    model.eq_data[cid, 0:3] = anchor1
    model.eq_data[cid, 3:6] = anchor2
    data.eq_active[cid] = 1


def check_connect_constraint(model, data, cid):
    if model.eq_active[cid] == 0:
        return 0
    id1 = model.eq_obj1id[cid]
    id2 = model.eq_obj2id[cid]
    anchor1 = data.xmat[id1].reshape(3, 3) @ model.eq_data[cid, 0:3] + data.xpos[id1]
    anchor2 = data.xmat[id2].reshape(3, 3) @ model.eq_data[cid, 3:6] + data.xpos[id2]
    res = np.linalg.norm(anchor1 - anchor2)
    return res


class CuRoboArm:
    def __init__(self,
                 robot_file: str = "franka.yml",
                 num_ik_seeds: int = 1,
                 device=torch.device("cuda", 0)
                 ):
        self.tensor_args = TensorDeviceType(device=device)
        robot_yaml = \
        load_yaml(join_path("model_lib/asset/franka_emika_panda", robot_file))[
            "robot_cfg"]
        self.robot_cfg = RobotConfig.from_dict(robot_yaml)
        self.world_collision = None
        self.kin_model = CudaRobotModel(self.robot_cfg.kinematics)
        self.pk_chain = pk.build_serial_chain_from_urdf(
            open(join_path(get_assets_path(),
                           robot_yaml['kinematics']['urdf_path'])).read(),
            "ee_link")
        self.pk_chain.to(dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        self.num_ik_seeds = num_ik_seeds
        self.curobo_fn = None  # for collision checking
        self.local_ik_solver = None  # for ik solving
        self.global_ik_solver = None  # for ik solving

    def update_world(self, scene_mesh: Meshes):
        scene_mesh_curobo = Mesh(
            name="scene",
            pose=[0, 0, 0, 1, 0, 0, 0],
            vertices=scene_mesh.verts_packed().detach().cpu().numpy().tolist(),
            faces=scene_mesh.faces_packed().detach().cpu().numpy().tolist())
        world_cfg = WorldConfig(mesh=[scene_mesh_curobo])
        # TODO: use the update_world instead of recreating
        robotworld_config = RobotWorldConfig.load_from_config(
            self.robot_cfg, world_cfg, collision_activation_distance=0.0
        )
        self.curobo_fn = RobotWorld(robotworld_config)
        self.world_collision = robotworld_config.world_model
        ik_config_kwargs = {
            "robot_cfg": self.robot_cfg,
            "world_model": None,  # reuse the WorldCollision
            "world_coll_checker": self.world_collision,
            "rotation_threshold": 0.1,
            "position_threshold": 0.01,
            "num_seeds": self.num_ik_seeds,
            "self_collision_check": True,
            "self_collision_opt": True,
            "collision_activation_distance": None,
            "tensor_args": self.tensor_args,
        }
        local_ik_config = IKSolverConfig.load_from_robot_config(
            **ik_config_kwargs,
            use_particle_opt=False,
            gradient_file="gradient_ik_controller.yml",
            use_cuda_graph=True,
        )
        global_ik_config = IKSolverConfig.load_from_robot_config(
            **ik_config_kwargs,
            use_cuda_graph=True,
        )
        self.local_ik_solver = IKSolver(local_ik_config)
        self.global_ik_solver = IKSolver(global_ik_config)

    def compute_ik_from_current_joint(self, goal, q_current: torch.tensor = None):
        # B goal, one or B q_current
        # if current joint angles are provided, use it as seeds
        seed_config = None
        retract_config = None
        B = len(goal)
        if q_current is not None:
            if q_current.dim() == 2:
                assert len(q_current) == B
                seed_config = q_current.repeat(self.num_ik_seeds, 1, 1)
            elif q_current.dim() == 1:
                seed_config = q_current.repeat(B, self.num_ik_seeds, 1)
            retract_config = seed_config[:, 0].clone()

        if q_current is None:
            result = self.global_ik_solver.solve_batch(goal)
        else:
            result = self.local_ik_solver.solve_batch(
                goal, seed_config=seed_config, retract_config=retract_config)
        # js = result.solution.squeeze()
        # if result.position_error > 0.005:
        #     print(f"Warning: IK solution error: {result.position_error}")
        return result

    def compute_collision_cost(self, q):
        d_world, d_self = self.curobo_fn.get_world_self_collision_distance_from_joints(
            q)
        return d_world + d_self

    def forward_kinematics(self, qs):
        if not torch.is_tensor(qs):
            qs = torch.tensor(qs, **self.tensor_args.as_torch_dict())
        return self.kin_model.get_state(qs)

    def compute_manipulability_cost(self, q):
        with torch.cuda.amp.autocast(enabled=False):
            jac = pk.jacobian.calc_jacobian(self.pk_chain, q)
            J_J_t = torch.matmul(jac, jac.transpose(-2, -1))
            score = torch.sqrt(torch.det(J_J_t))
        return score
