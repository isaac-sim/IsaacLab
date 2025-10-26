
import torch
import omni.physics.tensors.impl.api as physx
from isaacsim.core.prims import XFormPrim
from isaaclab.utils.math import convert_quat


def obtain_world_pose_from_view(
    physx_view: XFormPrim | physx.ArticulationView | physx.RigidBodyView,
    env_ids: torch.Tensor,
    clone: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the world poses of the prim referenced by the prim view.
    Args:
        physx_view: The prim view to get the world poses from.
        env_ids: The environment ids of the prims to get the world poses for.
    Returns:
        A tuple containing the world positions and orientations of the prims. Orientation is in wxyz format.
    Raises:
        NotImplementedError: If the prim view is not of the correct type.
    """
    if isinstance(physx_view, XFormPrim):
        pos_w, quat_w = physx_view.get_world_poses(env_ids)
    elif isinstance(physx_view, physx.ArticulationView):
        pos_w, quat_w = physx_view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = convert_quat(quat_w, to="wxyz")
    elif isinstance(physx_view, physx.RigidBodyView):
        pos_w, quat_w = physx_view.get_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = convert_quat(quat_w, to="wxyz")
    else:
        raise NotImplementedError(f"Cannot get world poses for prim view of type '{type(physx_view)}'.")

    if clone:
        return pos_w.clone(), quat_w.clone()
    else:
        return pos_w, quat_w