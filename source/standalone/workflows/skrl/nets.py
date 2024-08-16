import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

from transformers import AutoImageProcessor, ViTModel

DEBUG = False

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
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
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
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
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
        if DEBUG:
            print(f"{img.mean()}+-{img.std()}: [{img.min()}, {img.max()}]")
        joint_pos = x["joint_pos"]
        joint_vel = x["joint_vel"]
        last_action = x["last_action"]
        states = torch.cat([joint_pos, joint_vel, last_action], dim=1)
        linear_out = self.linear(states)
        cnn_out = self.cnn(img)
        out = torch.cat([linear_out, cnn_out], dim=1)
        return self.fc(out)
    

class FastAttention(nn.Module):
    def __init__(self, dim, heads=8, k=64):
        super(FastAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.k = k

        # Linear projections for queries, keys, and values
        self.to_q = nn.Linear(dim, heads * k, bias=False)
        self.to_k = nn.Linear(dim, heads * k, bias=False)
        self.to_v = nn.Linear(dim, heads * k, bias=False)

        # Output projection
        self.to_out = nn.Linear(heads * k, dim, bias=False)

    def forward(self, x):
        b, n, d = x.shape  # Batch size, sequence length, and embedding dimension
        h = self.heads

        # Linear projections
        q = self.to_q(x).view(b, n, h, self.k)
        k = self.to_k(x).view(b, n, h, self.k)
        v = self.to_v(x).view(b, n, h, self.k)

        # Softmax over keys (for normalization)
        k_softmax = torch.softmax(k, dim=1)

        # Compute the attention output
        # Efficient computation by rearranging operations
        qk = torch.einsum('bnhd,bnhe->bhde', q, k_softmax)
        qkv = torch.einsum('bhde,bnhe->bnhd', qk, v)

        # Concatenate heads and apply the output projection
        out = qkv.contiguous().view(b, n, h * self.k)
        return self.to_out(out)
    
class AttentionGRUCNNMixNet(nn.Module):
    def __init__(self, observation_space, img_space, rel_pos_space, rel_vel_space, last_action_space, hidden_dim=256):
        super(AttentionGRUCNNMixNet, self).__init__()

        self.hiden_dim = hidden_dim

        rel_pos_size = 9
        rel_vel_size = 9
        last_action_size = 8
        img_space_size = 921600

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.SiLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.SiLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.SiLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((2, 2)),
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
            self.atten = FastAttention(dim=sample_cnn_output.shape[1])
            sample_cnn_output = sample_cnn_output.view(sample_cnn_output.shape[0], -1)
            # (batch, input_dim)
            sample_cnn_output = self.atten(sample_cnn_output.unsqueeze(1)).squeeze(1)
            self.gru = nn.GRU(input_size=sample_cnn_output.shape[1], hidden_size=256)
            sample_cnn_output, _ = self.gru(sample_cnn_output)
            self.head = nn.Linear(sample_cnn_output.shape[1], 256)

            self.fc = nn.Sequential(
                nn.Linear(sample_linear_output.shape[-1] + sample_cnn_output.shape[-1], 512),
                nn.ELU(),
                nn.Linear(512, 16),
                nn.ELU(),
                nn.Linear(16, 64),
                nn.ELU(),
                nn.Linear(64, 64),
                nn.Tanh(),
            )

    def forward(self, x):
        img = x["cam_data"].float()
        joint_pos = x["joint_pos"]
        joint_vel = x["joint_vel"]
        last_action = x["last_action"]

        if DEBUG:
            print(f"{img.mean()}+-{img.std()}: [{img.min()}, {img.max()}]")
            print(f"{joint_pos.mean()}+-{joint_pos.std()}: [{joint_pos.min()}, {joint_pos.max()}]")
            print(f"{joint_vel.mean()}+-{joint_vel.std()}: [{joint_vel.min()}, {joint_vel.max()}]")
            print(f"{last_action.mean()}+-{last_action.std()}: [{last_action.min()}, {last_action.max()}]")
        
        states = torch.cat([joint_pos, joint_vel, last_action], dim=1)
        linear_out = self.linear(states)

        cnn_out = self.cnn(img)
        cnn_out = cnn_out.view(img.shape[0], -1)
        cnn_out = self.atten(cnn_out.unsqueeze(1)).squeeze(1)
        cnn_out, _ = self.gru(cnn_out)
        cnn_out = self.head(cnn_out)
        
        out = torch.cat([linear_out, cnn_out], dim=1)
        return self.fc(out)
    
class ResnetCNNMixNet(nn.Module):
    def __init__(self, observation_space, img_space, rel_pos_space, rel_vel_space, last_action_space):
        super(ResnetCNNMixNet, self).__init__()

        # Flat shapes
        # img_space_size = observation_space.spaces["cam_data"].shape
        # rel_pos_size = observation_space.spaces["joint_pos"].shape
        # rel_vel_size = observation_space.spaces["joint_vel"].shape
        # last_action_size = observation_space.spaces["actions"].shape

        rel_pos_size = 9
        rel_vel_size = 9
        last_action_size = 8
        img_space_size = 921600

        # NOTE: Pretrained backbone
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.backbone = resnet50(weights=self.weights)
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-6]))
        self.preprocess = self.weights.transforms()

        self.cnn = nn.Sequential(
            self.backbone,
            nn.Conv2d(64, 32, kernel_size=8, stride=4),
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
            sample_img = self.preprocess(sample_img)
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
        img = self.preprocess(img)
        if DEBUG:
            print(f"{img.mean()}+-{img.std()}: [{img.min()}, {img.max()}]")
        
        joint_pos = x["joint_pos"]
        joint_vel = x["joint_vel"]
        last_action = x["last_action"]
        states = torch.cat([joint_pos, joint_vel, last_action], dim=1)

        cnn_out = self.cnn(img)
        linear_out = self.linear(states)
        
        out = torch.cat([cnn_out, linear_out], dim=1)
        return self.fc(out)
    
class ViTMix(nn.Module):
    def __init__(self, observation_space, img_space, rel_pos_space, rel_vel_space, last_action_space, device):
        super(ViTMix, self).__init__()

        # self.weights = ViT_B_16_Weights.DEFAULT
        # self.backbone = vit_b_16(weights=self.weights)
        # self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
        # self.process = self.weights.transforms()

        rel_pos_size = 9
        rel_vel_size = 9
        last_action_size = 8
        img_space_size = 921600

        self.device = device
        self.process = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

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
            sample_img = self.process(sample_img, return_tensors="pt", do_rescale=False, use_fast=True)
            sample_img = self.backbone(**sample_img)
            sample_img = sample_img.last_hidden_state
            sample_img = sample_img.view(-1, sample_img.shape[1] * sample_img.shape[2])
            self.head = nn.Sequential(
                nn.Linear(sample_img.shape[-1], 64),
            )
            sample_img_output = self.head(sample_img)

            self.fc = nn.Sequential(
                nn.Linear(sample_linear_output.shape[-1] + sample_img_output.shape[-1], 512),
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
        img = self.process(img, return_tensors="pt", do_rescale=False, use_fast=True).to(self.device)
        if DEBUG:
            print(f"{img.mean()}+-{img.std()}: [{img.min()}, {img.max()}]")
        
        joint_pos = x["joint_pos"]
        joint_vel = x["joint_vel"]
        last_action = x["last_action"]
        states = torch.cat([joint_pos, joint_vel, last_action], dim=1)

        with torch.no_grad():
            cnn_out = self.backbone(**img)
            cnn_out = cnn_out.last_hidden_state

        cnn_out = cnn_out.view(-1, cnn_out.shape[1] * cnn_out.shape[2])
        cnn_out = self.head(cnn_out)
        linear_out = self.linear(states)
        
        out = torch.cat([cnn_out, linear_out], dim=1)
        return self.fc(out)

    
def test():
    import numpy as np
    model = ViTMix(None, None, None, None, None, None)
    sample = torch.from_numpy(np.zeros(shape=(1, 3, 480, 640)))
    output = model(sample)
    print(f"output: {output.shape}")

if __name__ == "__main__":
    test()