import torch
import torch.nn as nn
    
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, scandots_output_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)
        return latent
    

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, depth_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        num_prop = depth_cfg['num_prop']
        if num_prop == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + num_prop, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.recurrent_size = 512
            
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32+2),
                                last_activation
                            )
        self.hidden_states = torch.zeros(1, 0, 512)
        self.rnn.flatten_parameters()

    def forward(self, depth_image, proprioception):
        if self.hidden_states.shape[1] == 0:
            # On the first forward pass, initialize hidden states to proper batch size
            self.hidden_states = torch.zeros(1, depth_image.shape[0], self.recurrent_size).to(depth_image.device)

        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()
