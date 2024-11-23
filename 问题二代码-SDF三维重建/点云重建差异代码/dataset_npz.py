import glob
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import os
import trimesh
from joblib import Parallel, delayed
import numpy as np

class MyDataset(Dataset):
    def __init__(self, dir, device):
        self.dir = dir
        self.device = device
        self.files = [os.path.join(root, m)
                      for root, _, files in os.walk(self.dir)
                      for m in files if m.endswith('.npz')][0:50]

        print("\n".join(self.files))
        self.samples = []
        self.len = 0
        self.fileslen = len(self.files)
        def load_mesh(mesh_name):
            mesh_path = mesh_name
            numpy_data = np.load(mesh_path)
            samples = np.concatenate((numpy_data['pos'], numpy_data['neg']))
            samples = samples[~np.isnan(samples).any(axis=1)]   # 去除nan数据
            return mesh_name, samples

        # 加载npz文件到内存
        mesh_pairs = Parallel(n_jobs=-1, verbose=10)(delayed(load_mesh)(files_name) for files_name in self.files)
        for mesh_name, sample in mesh_pairs:
            sample_tensor = torch.from_numpy(sample).float()
            self.samples.append(sample_tensor)
        self.len = sum(len(sample) for sample in self.samples)
    def __len__(self):
        return self.len

    def get_files_len(self):
        return self.fileslen

    def __getitem__(self, idx):
        idx_file = 0
        while idx >= len(self.samples[idx_file]):
            idx -= len(self.samples[idx_file])
            idx_file += 1
        sample = self.samples[idx_file][idx]

        res = {
            'xyz': sample[:3],  # 已经是张量，可以直接使用
            'sdf': sample[3],
            'idx': torch.tensor(idx_file, device=self.device)
        }

        # 返回样本数据
        return res


# 测试数据类功能
from model.SDF import *
from torch.utils import data
if __name__ == '__main__':
    dataset = MyDataset('../dataset/SDFdata_train/SdfSamples/ShapeNetV2/02691156', 'cuda')
    print(dataset.get_files_len())
    print(dataset.__len__())
    # torch.set_printoptions(threshold=torch.inf)
    # embedding = LatentEmbedding(dataset.get_files_len(), 256).to('cuda')
    # Trainloader = data.DataLoader(dataset=dataset, batch_size=1024, shuffle=False, num_workers=0)
    # for i, data in enumerate(Trainloader):
    #     xyz = data['xyz']
    #     latent_idx = data['idx']
    #     sdf = data['sdf']
    #     latent = embedding(latent_idx)
    #     if torch.isnan(sdf).any():
    #         print("SDF contains NaN values.")
    #         print(i)
    #         torch.set_printoptions(threshold=torch.inf)
    #         print(f"sdf: {sdf}")  # sdf 可能是目标值，直接打印
    #         print(xyz)
    #         print(latent)
    #
    #         exit(-1)
    #     print('====')
    #     print(xyz)
    #     print(latent)
    #     print(sdf)