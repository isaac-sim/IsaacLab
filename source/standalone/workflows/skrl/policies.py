import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from nets import LinearNet, CNNNet, LargeCNNMixNet, CNNMixNet, ViTMix

DEBUG = False
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", type="linear"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.arch_type = type
        if self.arch_type == "linear":
            print(f"{self.arch_type.title()} architecture\n")
            self.net = LinearNet(self.num_observations).net
        elif self.arch_type == "cnn":
            print(f"{self.arch_type.title()} architecture\n")
            self.IMG_SHAPE = (480, 640, 3) # TODO: Make dynamic
            self.net = CNNNet(observation_space.shape, img_space=self.IMG_SHAPE).net
        elif self.arch_type == "cnn_mix":
            print(f"{self.arch_type.title()} architecture\n")
            self.REL_POS_SHAPE = (9) # 0 TODO: Make dynamic
            self.REL_VEL_SHAPE = (9) # 1 TODO: Make dynamic
            self.LAST_ACTION_SHAPE = (8) # 2 TODO: Make dynamic
            self.IMG_SHAPE = (480, 640, 3) # 3 TODO: Make dynamic
            # self.net = CNNMixNet(
            #     observation_space, 
            #     img_space=self.IMG_SHAPE,
            #     rel_pos_space = self.REL_POS_SHAPE,
            #     rel_vel_space = self.REL_VEL_SHAPE,
            #     last_action_space = self.LAST_ACTION_SHAPE,
            # )

            self.net = LargeCNNMixNet(
                observation_space, 
                img_space=self.IMG_SHAPE,
                rel_pos_space = self.REL_POS_SHAPE,
                rel_vel_space = self.REL_VEL_SHAPE,
                last_action_space = self.LAST_ACTION_SHAPE,
            )

            # self.net = ViTMix(
            #     observation_space, 
            #     img_space=self.IMG_SHAPE,
            #     rel_pos_space = self.REL_POS_SHAPE,
            #     rel_vel_space = self.REL_VEL_SHAPE,
            #     last_action_space = self.LAST_ACTION_SHAPE,
            # )
        else:
            raise NotImplementedError

        # Actions
        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        # Value
        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            inputs = inputs["states"]
            if self.arch_type == "cnn":
                inputs = inputs.view(-1, *self.IMG_SHAPE).permute(0, 3, 1, 2)
                inputs = inputs / inputs.max()
            elif self.arch_type == "cnn_mix":
                joint_pos_input = inputs[:, :9]
                joint_vel_input = inputs[:, 9:18]
                last_action_input = inputs[:, 18:26]
                img_input = inputs[:, 26:]
                inputs = {
                    "cam_data": img_input,
                    "joint_pos": joint_pos_input,
                    "joint_vel": joint_vel_input,
                    "last_action": last_action_input
                }
                
                img_input = inputs["cam_data"].view(-1, *self.IMG_SHAPE).permute(0, 3, 1, 2)
                img_input = img_input / img_input.max()
                joint_pos_input = inputs["joint_pos"].view(-1, self.REL_POS_SHAPE)
                joint_vel_input = inputs["joint_vel"].view(-1, self.REL_VEL_SHAPE)
                last_action_input = inputs["last_action"].view(-1, self.LAST_ACTION_SHAPE)

                inputs = {
                    "cam_data": img_input,
                    "joint_pos": joint_pos_input,
                    "joint_vel": joint_vel_input,
                    "last_action": last_action_input
                }

            self._shared_output = self.net(inputs)
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        
        elif role == "value":
            inputs = inputs["states"]
            if self.arch_type == "cnn":
                inputs = inputs.view(-1, *self.IMG_SHAPE).permute(0, 3, 1, 2)
                inputs = inputs / inputs.max()
            elif self.arch_type == "cnn_mix":
                joint_pos_input = inputs[:, :9]
                joint_vel_input = inputs[:, 9:18]
                last_action_input = inputs[:, 18:26]
                img_input = inputs[:, 26:]
                inputs = {
                    "cam_data": img_input,
                    "joint_pos": joint_pos_input,
                    "joint_vel": joint_vel_input,
                    "last_action": last_action_input
                }

                img_input = inputs["cam_data"].view(-1, *self.IMG_SHAPE).permute(0, 3, 1, 2)
                img_input = img_input / img_input.max()
                joint_pos_input = inputs["joint_pos"].view(-1, self.REL_POS_SHAPE)
                joint_vel_input = inputs["joint_vel"].view(-1, self.REL_VEL_SHAPE)
                last_action_input = inputs["last_action"].view(-1, self.LAST_ACTION_SHAPE)
                
                inputs = {
                    "cam_data": img_input,
                    "joint_pos": joint_pos_input,
                    "joint_vel": joint_vel_input,
                    "last_action": last_action_input
                }

                
            shared_output = self.net(inputs) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}
        