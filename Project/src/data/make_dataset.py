import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import Subset
import os
import pandas as pd
from PIL import Image

torch.manual_seed(0)
torch.random.manual_seed(0)


class CreateDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform):
        self.train_df = pd.read_csv(csv_path, skipinitialspace = True, delimiter = "\t")
        self.root_dir = root_dir
        self.images = os.listdir(self.root_dir)
        self.transform = transform
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        # 画像の読み込み
        image_name = self.images[idx]
        image = Image.open(os.path.join(self.root_dir, image_name))
        
        # ラベルの読み混み
        label = self.train_df.query('file_name=="'+image_name+'"')['label_id'].iloc[0]
        
        return self.transform(image), int(label)

# 前処理方法の指定
transform = transforms.Compose([
    transforms.ToTensor()
])

# フォルダからデータ読み込み
train_val_set = CreateDataset("../../../Data/train_master.tsv", "../../../Data/train", transform)


# dataset読みこみ
n_samples = len(train_val_set)

train_size = n_samples * 0.9


subset1_indices = range(0,int(train_size)) 
subset2_indices = range(int(train_size),n_samples) 

trainset = Subset(train_val_set, subset1_indices)
valset   = Subset(train_val_set, subset2_indices)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True)

valloader = torch.utils.data.DataLoader(valset, batch_size=len(valset), shuffle=False)


