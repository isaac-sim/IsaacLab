from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as T
import torch
import ast 
import re

class CubeDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # For HPC
        # imgA = Image.open('../../../../Linuka_cube_prediction_dataset/' + row['front_img_rgb']).convert('RGB')
        # imgB = Image.open('../../../../Linuka_cube_prediction_dataset/' + row['side_img_rgb']).convert('RGB')
        # imgC = Image.open('../../../../Linuka_cube_prediction_dataset/' + row['bird_img_rgb']).convert('RGB')
        

        imgA = Image.open('../../../hri-pl-frm-mvvd/data/mounted_dataset/work/Linuka_cube_prediction_dataset/' + row['front_img_rgb']).convert('RGB')
        imgB = Image.open('../../../hri-pl-frm-mvvd/data/mounted_dataset/work/Linuka_cube_prediction_dataset/' + row['side_img_rgb']).convert('RGB')
        imgC = Image.open('../../../hri-pl-frm-mvvd/data/mounted_dataset/work/Linuka_cube_prediction_dataset/' + row['bird_img_rgb']).convert('RGB')

        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
            imgC = self.transform(imgC)

        # Replace any whitespace between numbers with a comma
        def fix_list_string(s):
            return re.sub(r'(?<=\d)\s+(?=[-\d])', ', ', s)

        pos_str = fix_list_string(row['cube_changed_pos'])
        #ore_str = fix_list_string(row['cube_changed_ore'])
        pos = ast.literal_eval(pos_str)
        #ore = ast.literal_eval(ore_str)
        target = torch.tensor(pos, dtype=torch.float32)
        return imgA, imgB, imgC, target
