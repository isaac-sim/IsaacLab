import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities.model_helpers import _ModuleMode

from configs.user_config import BASE_PATH
from datasets.diffusion_reach_dataset import DiffusionSubgoalDataset

from dis_learning.dis_net import DisNet

from model_lib.hinge_model import HingeModel
from model_lib.rope_model import RopeModel
from utils.data_utils import find_best_checkpoint, update_config


class ModuleMode:
    def __init__(self, model):
        self._module_mode = _ModuleMode()
        self._module_mode.capture(model)

    def restore(self, model):
        self._module_mode.restore(model)


def load_subgoal_diffuser(diff_model_path, cfg=None):
    from reachability.diff_reachnet import DiffusionReachNet
    diff_cfg = OmegaConf.load(os.path.join(BASE_PATH, diff_model_path, 'config.yaml'))
    if cfg is not None:
        # diff_cfg.update(cfg)
        diff_cfg.dis_logsum_temp = cfg.dis_logsum_temp
    diff_model = DiffusionReachNet(diff_cfg)
    model_dict = torch.load(find_best_checkpoint(diff_model_path))
    diff_model.load_state_dict(model_dict['state_dict'], strict=False)
    diff_model.cuda()
    diff_model.eval()
    return diff_model


def load_disnet_model(disnet_path, cfg):
    disnet_cfg = OmegaConf.load(os.path.join(BASE_PATH, disnet_path, 'config.yaml'))
    if "dis_logsum_temp" in cfg:
        disnet_cfg.logsum_temp = cfg['dis_logsum_temp']
    disnet_model = DisNet(disnet_cfg)
    model_dict = torch.load(find_best_checkpoint(disnet_path))
    disnet_model.load_state_dict(model_dict['state_dict'])
    disnet_model.eval()
    disnet_model.cuda()
    return disnet_model


def load_ndf_model(ndf_path, cfg=None):
    from latent_subgoals.implicit_ndf import ImplicitNDF
    ndf_cfg = OmegaConf.load(os.path.join(BASE_PATH, ndf_path, 'config.yaml'))
    if cfg is not None:
        ndf_cfg.ndf_normalize_feat = cfg.get("ndf_normalize_feat", True)
        if "ndf_acts" in cfg:
            ndf_cfg.acts = cfg.ndf_acts
    ndf_model = ImplicitNDF(ndf_cfg)
    model_dict = torch.load(find_best_checkpoint(ndf_path))
    new_state_dict = {}
    for k, v in model_dict['state_dict'].items():
        if "model.decoder.fc_out" in k:
            k = k.replace("model.decoder.fc_out", "model.decoder.inv_fc_out")
        new_state_dict[k] = v
    ndf_model.load_state_dict(new_state_dict)

    ndf_model.eval()
    ndf_model.cuda()
    ndf_model.requires_grad_(False)
    return ndf_model


def load_point_vae_model(point_vae_path, cfg=None, load_ae=False):
    from latent_subgoals.point_vae import PointVAE
    model_dict = torch.load(find_best_checkpoint(point_vae_path))
    hyper_params = model_dict['hyper_parameters']
    if cfg is not None:
        # assert cfg.ndf_acts == hyper_params["cfg"]["ndf_acts"]
        if "dis_logsum_temp" in cfg:
            print("dis_logsum_temp is updated to ", cfg.dis_logsum_temp)
            hyper_params["cfg"]["dis_logsum_temp"] = cfg.dis_logsum_temp

    hyper_params["cfg"]["is_deterministic"] = True
    point_vae = PointVAE(**hyper_params)
    point_vae.load_state_dict(model_dict['state_dict'], strict=False)
    point_vae.eval()
    point_vae.cuda()
    point_vae.requires_grad_(False)
    if load_ae:
        point_ae = point_vae.model
        del point_vae
        return point_ae
    return point_vae


def create_dyn_model(cfg):
    cfg.num_plan_worker = cfg.get("num_plan_worker", 0)
    cfg.rope_stiffness = cfg.get("rope_stiffness", 0.1)
    if "rope" in cfg.env_type:
        dyn_model = RopeModel(num_worker=cfg.num_plan_worker,
                              rope_stiffness=cfg.rope_stiffness,
                              env_type=cfg.env_type,
                              )
    elif cfg.env_type == "hinge":
        dyn_model = HingeModel(state_space=cfg.state_space,
                               num_worker=cfg.num_plan_worker,
                               cam_name=cfg.get("cam_name", "rgb"),
                               )
    return dyn_model


def prepare_model(cfg):
    dyn_model = create_dyn_model(cfg)
    if cfg.run_mode == "opt_plan":
        model_cfg = OmegaConf.load(os.path.join(BASE_PATH, cfg.model_path, 'config.yaml'))
        cfg = update_config(model_cfg, cfg)
        # cfg.ds_dir = os.path.join(cfg.base_path, 'dataset', cfg.ds_dir)
    cfg.base_path = BASE_PATH
    cfg.goal_cache_path = os.path.join(cfg.base_path, cfg.goal_cache_path)

    subgoal_model = None
    ndf_model = None

    if cfg["run_mode"] == "oracle_opt_plan":
        # if cfg["run_mode"] == "opt_plan_oracle":
        from reachability.oracle_subgoal_model import OracleSubgoalModel
        subgoal_model = OracleSubgoalModel(cfg, env=dyn_model.env)
        train_ds = DiffusionSubgoalDataset(cfg, 'val')
        subgoal_model.ds = train_ds
    elif cfg["run_mode"] == "opt_plan":
        subgoal_model = load_subgoal_diffuser(cfg.model_path, cfg)
        if cfg.expensive_vis:
            train_ds = DiffusionSubgoalDataset(subgoal_model.cfg, 'val')
            subgoal_model.train_ds = train_ds
        # if subgoal_model.cfg.get("dis_model_path", None) and subgoal_model.diff_mode == "chain_hier":
        #     subgoal_model.dis_model = load_disnet_model(cfg.dis_model_path, cfg)
    else:
        from reachability.dummy_subgoal_model import DummySubgoalModel
        subgoal_model = DummySubgoalModel(cfg)

    if "rope" in cfg.env_type:
        noise_sigma = torch.eye(3, dtype=torch.float64) * cfg.mppi_sigma
    elif cfg.env_type == "hinge":
        noise_sigma = torch.eye(3, dtype=torch.float64) * cfg.mppi_sigma
        noise_sigma[1, 1] = 1e-5
    return dyn_model, subgoal_model, noise_sigma, cfg
