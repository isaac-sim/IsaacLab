import warp as wp

@wp.kernel
def concat_pos_and_quat_to_pose_kernel(
    pos: wp.array2d(dtype=wp.vec3f),
    quat: wp.array2d(dtype=wp.quatf),
    pose: wp.array2d(dtype=wp.transformf),
):
    """Concatenate position and quaternion to pose.

    Args:
        pos: Position array. Shape is (N, B).
        quat: Quaternion array. Shape is (N, B).
        pose: Pose array. Shape is (N, B).
    """
    env, sensor = wp.tid()
    pose[env, sensor] = wp.transform(pos[env, sensor], quat[env, sensor])