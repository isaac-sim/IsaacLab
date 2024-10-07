import numpy as np
import torch
from torch.utils.data import Dataset
from source.utils.data_utils import read_h5_dict


class CubeSlipDataset(Dataset):
    def __init__(self):
        self.all_trajs = read_h5_dict("data.h5")
        self.num_states = self.all_trajs["action"].shape[0]
    
    def __len__(self):
        return self.num_states
    
    def __getitem__(self, idx):
        cur_state = torch.tensor(self.all_trajs["cur_state"][idx], dtype=torch.float32)
        action = torch.tensor(self.all_trajs["action"][idx], dtype=torch.float32)
        is_slip = torch.tensor(self.all_trajs["is_slip"][idx], dtype=torch.float32)
        
        x = torch.cat((cur_state, action), dim=0)
        data = {
            "x": x,
            "y": is_slip
        }
        
        return data