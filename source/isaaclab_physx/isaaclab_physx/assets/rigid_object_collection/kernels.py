import warp as wp

@wp.kernel
def resolve_view_ids(
    env_ids: wp.array(dtype=wp.int32),
    body_ids: wp.array(dtype=wp.int32),
    view_ids: wp.array(dtype=wp.int32),
    num_query_envs: wp.int32,
    num_total_envs: wp.int32,
) -> None:
    i, j = wp.tid()
    view_ids[j * num_query_envs + i] = body_ids[j] * num_total_envs + env_ids[i]