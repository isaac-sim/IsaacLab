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

            self._shared_output = self.net(inputs)
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        
        elif role == "value":
            inputs = inputs["states"]
            if self.arch_type == "cnn":
                inputs = inputs.view(-1, *self.IMG_SHAPE).permute(0, 3, 1, 2)
                
            shared_output = self.net(inputs) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}
        