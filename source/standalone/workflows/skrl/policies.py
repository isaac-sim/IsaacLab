import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

class LinearNet(nn.Module):
    def __init__(self, input_features):
        super(LinearNet, self).__init__()

        self.net = nn.Sequential(nn.Linear(input_features, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())
        
    def forward(self, x):
        return self.net(x)
    
class CNNNet(nn.Module):
    def __init__(self, observation_space, img_space):
        super(CNNNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_x = torch.zeros(observation_space)[None, ...].view(-1, *img_space).permute(0, 3, 1, 2)
            sample_output = self.cnn(sample_x)

            self.fc = nn.Sequential(
                nn.Linear(sample_output.shape[-1], 512),
                nn.ReLU(),
                nn.Linear(512, 16),
                nn.Tanh(),
                nn.Linear(16, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )

        self.net = nn.Sequential(self.cnn, self.fc)

    def forward(self, x):
        return self.net(x)
    

class CNNMixNet(nn.Module):
    def __init__(self, observation_space, img_space, rel_pos_space, rel_vel_space, last_action_space):
        super(CNNMixNet, self).__init__()

        # Flat shapes
        # img_space_size = observation_space.spaces["cam_data"].shape
        # rel_pos_size = observation_space.spaces["joint_pos"].shape
        # rel_vel_size = observation_space.spaces["joint_vel"].shape
        # last_action_size = observation_space.spaces["actions"].shape

        rel_pos_size = 9
        rel_vel_size = 9
        last_action_size = 8
        img_space_size = 921600

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_rel_pos = torch.zeros(rel_pos_size)[None, ...].view(-1, rel_pos_space)
            sample_rel_vel = torch.zeros(rel_vel_size)[None, ...].view(-1, rel_vel_space)
            sample_last_action = torch.zeros(last_action_size)[None, ...].view(-1, last_action_space)
            sample_states = torch.cat([sample_rel_pos, sample_rel_vel, sample_last_action], dim=1)
            self.linear = nn.Sequential(
                nn.Linear(sample_states.shape[-1], 256),
                nn.ELU(),
                nn.Linear(256, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU()
            )
            sample_linear_output = self.linear(sample_states)
        
            sample_img = torch.zeros(img_space_size)[None, ...].view(-1, *img_space).permute(0, 3, 1, 2)
            sample_cnn_output = self.cnn(sample_img)

            self.fc = nn.Sequential(
                nn.Linear(sample_linear_output.shape[-1] + sample_cnn_output.shape[-1], 512),
                nn.ReLU(),
                nn.Linear(512, 16),
                nn.Tanh(),
                nn.Linear(16, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )

    def forward(self, x):
        img = x["cam_data"].float()
        joint_pos = x["joint_pos"]
        joint_vel = x["joint_vel"]
        last_action = x["last_action"]
        states = torch.cat([joint_pos, joint_vel, last_action], dim=1)
        linear_out = self.linear(states)
        cnn_out = self.cnn(img)
        out = torch.cat([linear_out, cnn_out], dim=1)
        return self.fc(out)



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
            self.net = CNNMixNet(
                observation_space, 
                img_space=self.IMG_SHAPE,
                rel_pos_space = self.REL_POS_SHAPE,
                rel_vel_space = self.REL_VEL_SHAPE,
                last_action_space = self.LAST_ACTION_SHAPE,
            )
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
        