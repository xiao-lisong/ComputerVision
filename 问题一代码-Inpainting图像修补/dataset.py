import glob
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import nibabel as nib

class MyDataset(Dataset):
    def __init__(self, imageSize, path):
        self.img_dir = path
        self.transform = transforms.Compose([transforms.Resize(imageSize),
                                    transforms.CenterCrop(imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.img_list = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])

        image = Image.open(img_path)

        data = self.transform(image)

        return data