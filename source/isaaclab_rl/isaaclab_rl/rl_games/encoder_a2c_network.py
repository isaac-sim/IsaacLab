from rl_games.algos_torch.network_builder import A2CBuilder
from rl_games.algos_torch.running_mean_std import RunningMeanStd

import torch
import torch.nn as nn

from .encoder_networks import Permute
from . import encoder_networks as encoders

from copy import deepcopy


class A2CBuilderWithEncoders(A2CBuilder):
    def load(self, params):
        self.params = params

    class Network(A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            raw_input_shape = kwargs.get('input_shape')
            enc_cfg = params.pop("encoders", None)

            # Build encoder stacks but DO NOT register yet (no ModuleDict before super)
            key_orders = list(raw_input_shape.keys())  # preserve concat order
            input_shape_cp = deepcopy(raw_input_shape)
            encoders_dict, encoder_group = {}, {}
            if enc_cfg:
                for enc_name, enc_conf in enc_cfg["encoder_cfgs"].items():
                    ops = []
                    groups = enc_conf["encoding_groups"]
                    first_grp = groups[0]

                    dummy_in = torch.zeros(1, *input_shape_cp[first_grp])
                    if enc_conf.get("transpose", False):
                        ops.append(Permute(0, 3, 1, 2))
                        dummy_in = ops[-1](dummy_in)
                    if enc_conf.get("normalize", False):
                        ops.append(RunningMeanStd(dummy_in.shape[1:]))

                    cls = getattr(encoders, enc_conf["class_type"])
                    ops.append(cls(dummy_in.shape[1:], **enc_conf))

                    encoders_dict[enc_name] = nn.Sequential(*ops)
                    with torch.no_grad():
                        out = encoders_dict[enc_name](torch.zeros(1, *input_shape_cp[first_grp]))

                    for grp in groups:
                        input_shape_cp[grp] = (out.size(1),)
                    encoder_group[enc_name] = groups
            kwargs["input_shape"] = (sum(s[0] for s in input_shape_cp.values()),)

            # Build the parent network first
            super().__init__(params, **kwargs)

            # Now it's safe to register modules
            self.encoders = nn.ModuleDict(encoders_dict)  # <- attach after super()
            self.encoder_group = encoder_group
            self._key_orders = key_orders

        def forward(self, obs_dict):
            # CAUTION:
            # The key ordering is consistent return in rl-games wrapper, but somehow RL games will preserve key order
            # in consistent order, it oscillates between iterations, we use _key_orders to ensure consistent ordering.
            for enc_name, encoder in self.encoders.items():
                for obs_key in self.encoder_group[enc_name]:
                    x = obs_dict['obs'][obs_key]
                    obs_dict['obs'][obs_key] = encoder(x)
            obs_dict['obs'] = torch.cat([obs_dict['obs'][key] for key in self._key_orders], dim=1)
            # delegate to parent's forward (which now sees only flat features)
            return super().forward(obs_dict)

    def build(self, name, **kwargs):
        return A2CBuilderWithEncoders.Network(self.params, **kwargs)
