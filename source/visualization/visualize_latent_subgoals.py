import copy

import numpy as np
import torch
from einops import rearrange, repeat

from latent_subgoals.latent_state_transform import latent_state_trans_func
from utils.loss_utils import compute_pts_dis
from utils.data_utils import AttrDict
from planning.mppi import _ensure_non_zero
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion, quaternion_multiply
from visualization.plot import plot_pointcloud
import torch.optim as optim

def compute_dis_latent_space(coords, target, batch, state_space,
                             ndf_model, point_ae, target_state=None, decoded_latent=False,
                             dis_func=None, scene_latent=None, smpl_pc_type="smpl_pc",requires_grad=False
                             ):
    """
    [] are potential targets for cost
    latent_ndf_pc
        target (subgoal):
           smpl_pc -> ndf_pc (ndf_model) -> [latent_ndf_pc] (point_ae.encoder) -> [ndf_pc] (point_ae.decoder)
        sampled (rollout):
          smpl_pc -> [ndf_pc] (ndf_model) -> [latent_ndf_pc] (point_ae.encoder)

    latent_pc
        target (subgoal):
           smpl_pc -> [latent_pc] (point_ae.encoder) -> [ndf_pc] (point_ae.decoder)
        smpl_pc (rollout):
          1. smpl_pc -> [latent_pc] (point_ae.encoder) -> [ndf_pc] (point_ae)
          2. smpl_pc -> [latent_pc] (point_ae.encoder)  and smpl_pc-> [ndf_pc] (ndf_model)

    """
    assert target is not None or target_state is not None

    # Compute sampled state
    batch[smpl_pc_type] = coords[None]
    batch["state"] = {smpl_pc_type: batch[smpl_pc_type]} # TODO: fix this hardcode
    # batch["state"]["smpl_pc2"] requires_grad is True
    if requires_grad:
        state_dict = latent_state_trans_func(batch,
                                             ndf_model=ndf_model, point_ae=point_ae,
                                             state_space=state_space,
                                             decoded_latent=decoded_latent,
                                             use_gt_ndf=True,
                                             scene_latent=scene_latent,
                                             smpl_pc_type=smpl_pc_type,
                                             requires_grad=requires_grad
                                             )
    else:
        with torch.no_grad():
            state_dict = latent_state_trans_func(batch,
                                             ndf_model=ndf_model, point_ae=point_ae,
                                             state_space=state_space,
                                             decoded_latent=decoded_latent,
                                             use_gt_ndf=True,
                                             scene_latent=scene_latent,
                                             smpl_pc_type=smpl_pc_type,
                                             requires_grad=requires_grad
                                             )
    sampled_state_dict = state_dict[0]

    # Compute target state
    if target_state is None:
        batch[smpl_pc_type] = target[None, None]
        batch["state"] = {smpl_pc_type: batch[smpl_pc_type]}
        if requires_grad:
            target_state_dict = latent_state_trans_func(batch,
                                                        ndf_model=ndf_model, point_ae=point_ae,
                                                        state_space=state_space,
                                                        decoded_latent=decoded_latent,
                                                        scene_latent=scene_latent,
                                                        smpl_pc_type=smpl_pc_type,
                                                        requires_grad=requires_grad
                                                        )[0]
        else:    
            with torch.no_grad():
                target_state_dict = latent_state_trans_func(batch,
                                                        ndf_model=ndf_model, point_ae=point_ae,
                                                        state_space=state_space,
                                                        decoded_latent=decoded_latent,
                                                        scene_latent=scene_latent,
                                                        smpl_pc_type=smpl_pc_type,
                                                        requires_grad=requires_grad
                                                        )[0]
    else:
        target_state_dict = target_state
    pair_dis = dis_func(sampled_state_dict, target_state_dict)
    return pair_dis


def sample_perturbed_state(raw_states, trans_noise=0.05, rad_noise=0.1, num_samples=200):
    # raw_states B x N
    # 3 trans + 4 quat + 2 radians
    B = raw_states.shape[0]
    d = raw_states.device
    sampled_states = repeat(raw_states, "b n -> b h n", h=num_samples).clone()
    sampled_states[..., :3] = sampled_states[..., :3] + torch.randn_like(sampled_states[..., :3]) * trans_noise
    random_euler = torch.randn((B, num_samples, 3), device=d) * rad_noise
    random_quat = matrix_to_quaternion(euler_angles_to_matrix(random_euler, "XYZ"))
    sampled_states[..., 3:7] = quaternion_multiply(random_quat, sampled_states[..., 3:7])
    sampled_states[..., 7:] = sampled_states[..., 7:] + torch.randn_like(sampled_states[..., 7:]) * rad_noise

    sampled_states[:, 0] = raw_states  # original states
    return sampled_states


def visualize_nearest_states(
        target,  # N subgoals in the form of smpl_pc
        scene_pc,
        dataset,
        ndf_model=None,
        point_ae=None,
        target_state=None,
        env=None,
        state_space="smpl_pc",
        decoded_latent=False,
        mode="sampling",
        bs=5,
        trials=100,
        trans_noise=0.05,
        rad_noise=0.1,
        vis_num=10,
        num_samples=200,
        adaptive_lambda=False,
        dis_func=None,
        scene_latent=None,
        smpl_pc_type="smpl_pc"
):
    total_num_traj = dataset.num_traj
    traj_len = dataset.traj_len
    data = dataset.ds_dict
    d = scene_pc.device

    if mode == "sampling":
        # sample random data from train_ds and compute the dis

        num_batches = total_num_traj // bs

        all_costs = []
        all_index = []
        all_noises = []
        for i in range(trials):
            bs_idx = torch.randint(0, num_batches, (1,)).item()
            sample_idx = np.arange(bs_idx * bs * traj_len, (bs_idx + 1) * bs * traj_len)
            random_trans = torch.randn((bs * traj_len, 1, 3)).to(d) * trans_noise
            coords = torch.tensor(data[smpl_pc_type][sample_idx]).to(d)
            coords += random_trans

            canon_pc = torch.tensor(data[smpl_pc_type][0], device=d)[None]
            batch = {"scene_pc": scene_pc[None], "canon_pc": canon_pc}
            pair_dis = compute_dis_latent_space(
                coords, target, batch, state_space,
                ndf_model, point_ae, target_state=target_state,
                decoded_latent=decoded_latent, dis_func=dis_func,
                scene_latent=scene_latent, smpl_pc_type=smpl_pc_type)
            # pair_dis = pair_dis[:, 0]
            all_index.append(sample_idx)
            all_costs.append(pair_dis)
            all_noises.append(random_trans)
        all_costs = torch.cat(all_costs)
        all_index = np.concatenate(all_index)
        all_noises = torch.cat(all_noises)

        if all_costs.shape[1] == 1:
            all_costs = all_costs.squeeze()
            sorted_ind = torch.argsort(all_costs)
            max_quantile = 0.001
            max_num = max(int(max_quantile * len(all_costs)), vis_num)
            selected_id = sorted_ind[torch.arange(vis_num).long()]
            # selected_id = sorted_ind[torch.linspace(0, max_num, vis_num).long()]
            costs = all_costs[selected_id]

            pcs = torch.tensor(data[smpl_pc_type][all_index[selected_id.detach().cpu()]], device=d) + all_noises[
                selected_id]
        else:
            sorted_ind = torch.argsort(all_costs, dim=0)
            costs = all_costs[sorted_ind[0], torch.arange(sorted_ind.shape[1])]
            pcs = torch.tensor(data[smpl_pc_type][all_index[sorted_ind[0].detach().cpu()]], device=d) + all_noises[
                sorted_ind[0]]

        return costs, pcs

    elif mode == "mppi":
        # sample bs initial states
        sample_idx = torch.randint(0, total_num_traj * traj_len, (vis_num,))
        raw_states = torch.tensor(data["raw_state"][sample_idx, :9]).to(d)
        ema_cost_mag = 0

        for i in range(trials):
            # sample perturbations
            sampled_raw_states = sample_perturbed_state(raw_states,
                                                        trans_noise=trans_noise,
                                                        rad_noise=rad_noise,
                                                        num_samples=num_samples)
            flat_raw_states = rearrange(sampled_raw_states, "b h n -> (b h) n")
            coords = env.transform_state(flat_raw_states, src="raw_state", tgt=smpl_pc_type)
            canon_pc = torch.tensor(data[smpl_pc_type][0], device=d)[None]
            # compute the cost for each perturbation
            batch = {"scene_pc": scene_pc[None], "canon_pc": canon_pc}
            pair_dis = compute_dis_latent_space(
                coords, target, batch, state_space,
                ndf_model, point_ae, target_state=target_state, decoded_latent=decoded_latent,
                dis_func=dis_func, scene_latent=scene_latent, smpl_pc_type=smpl_pc_type
            )
            # smpl_pc
            # mppi_lambda = 1e-3
            # ndf_occ last
            # mppi_lambda = 1e-3
            # ndf_occ last
            # mppi_lambda = 1e-3
            # # ndf_sdf all
            mppi_lambda = 1e-6

            sample_cost = rearrange(pair_dis, "(b h) 1 -> b h", b=vis_num)
            beta = torch.min(sample_cost, dim=1, keepdim=True)[0]
            cost_mag = torch.median(sample_cost).abs()
            if ema_cost_mag < 1e-5:
                ema_cost_mag = cost_mag
            else:
                ema_cost_mag = 0.9 * ema_cost_mag + 0.1 * cost_mag
            factor = 1 / mppi_lambda
            if adaptive_lambda:
                factor /= (ema_cost_mag + 1e-6)
            cost_non_zero = _ensure_non_zero(sample_cost, beta, factor)
            eta = torch.sum(cost_non_zero, dim=1, keepdim=True)
            omega = (1 / eta) * cost_non_zero
            # take the weighted averaged of the perturbations like mppi
            raw_states = torch.sum(omega.unsqueeze(-1) * sampled_raw_states, dim=1)
            # print(i, sample_cost.min())
        coords = env.transform_state(raw_states, src="raw_state", tgt=smpl_pc_type)
        all_costs = compute_dis_latent_space(
            coords, target, batch, state_space,
            ndf_model, point_ae, decoded_latent=decoded_latent,
            dis_func=dis_func, scene_latent=scene_latent, smpl_pc_type=smpl_pc_type).squeeze()
        sorted_ind = torch.argsort(all_costs)
        all_costs = all_costs[sorted_ind]
        coords = coords[sorted_ind]
        return all_costs, coords
    elif mode =="grad":
        # sample random initializations
        sample_idx = torch.randint(0, total_num_traj * traj_len, (vis_num,))
        raw_states = torch.tensor(data["raw_state"][sample_idx, :9], dtype=torch.float32, requires_grad=True,device=d)
        optimizer = optim.AdamW([raw_states], lr=1e-2, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000, eta_min=1e-3)
        for i in range(trials):
            optimizer.zero_grad()
            # evaluate distance to goal
            coords = env.transform_state(raw_states, src="raw_state", tgt=smpl_pc_type)
            canon_pc = torch.tensor(data[smpl_pc_type][0], device=d)[None]
            batch = {"scene_pc": scene_pc[None], "canon_pc": canon_pc}
            pair_dis = compute_dis_latent_space(
                coords, target, batch, state_space,
                ndf_model, point_ae, target_state=target_state, decoded_latent=decoded_latent,
                dis_func=dis_func, scene_latent=scene_latent, smpl_pc_type=smpl_pc_type,requires_grad=True
            )
            loss=pair_dis.mean()
            loss.backward()
            # add_gradient_noise([raw_states])
            optimizer.step()
            scheduler.step()
            print(f"Trial {i}: ",round(loss.cpu().detach().item(),3), "Min: ",round(pair_dis.min().cpu().detach().item(),3), " learning rate: ",scheduler.get_last_lr())
        coords = env.transform_state(raw_states, src="raw_state", tgt=smpl_pc_type)
        all_costs = compute_dis_latent_space(
            coords, target, batch, state_space,
            ndf_model, point_ae, decoded_latent=decoded_latent,
            dis_func=dis_func, scene_latent=scene_latent, smpl_pc_type=smpl_pc_type).squeeze()
        sorted_ind = torch.argsort(all_costs)
        all_costs = all_costs[sorted_ind]
        coords = coords[sorted_ind]
        return all_costs, coords

def visualize_nearest_states_parallel_targets(
        target,  # N subgoals in the form of smpl_pc
        scene_pc,
        dataset,
        ndf_model=None,
        point_ae=None,
        target_state=None,
        env=None,
        state_space="smpl_pc",
        decoded_latent=False,
        mode="sampling",
        bs=5,
        trials=100,
        trans_noise=0.05,
        rad_noise=0.1,
        vis_num=10,
        num_samples=200,
        adaptive_lambda=False,
        dis_func=None,
        scene_latent=None,
        smpl_pc_type="smpl_pc"
):
    total_num_traj = dataset.num_traj
    traj_len = dataset.traj_len
    data = dataset.ds_dict
    d = scene_pc.device
    return None