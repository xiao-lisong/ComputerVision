import glob
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import os
import trimesh
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
class MyDataset(Dataset):
    def __init__(self, dir, device):
        self.dir = dir
        self.device = device
        self.files = [m for m in os.listdir(self.dir) if m.endswith('.xlsx')]
        print("\n".join(self.files))

        self.meshes_df = {}
        self.len = 0
        self.fileslen = len(self.files)
        def load_mesh(mesh_name):
            return mesh_name, pd.read_excel(os.path.join(self.dir, mesh_name))

        # 加载训练形状文件到内存中
        mesh_pairs = Parallel(n_jobs=-1, verbose=10)(delayed(load_mesh)(files_name) for files_name in self.files)
        for mesh_name, df in mesh_pairs:
            self.meshes_df[mesh_name] = df

        self.len = sum(len(df) for df in self.meshes_df.values())
    
    # 返回样本点的数量
    def __len__(self):
        return self.len

    # 返回文件数 用于latent输入
    def get_files_len(self):
        return self.fileslen

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mesh_idx = idx // 50000
        sample_idx = idx % 50000

        samples_df = self.meshes_df[self.files[mesh_idx]]
        x = samples_df['x'][sample_idx]
        y = samples_df['y'][sample_idx]
        z = samples_df['z'][sample_idx]
        sdf = samples_df['sdf'][sample_idx]

        # 返回坐标 sdf 和形状序号
        sample = {
            'xyz': torch.from_numpy(np.array([x, y, z])).float(),
            'sdf': torch.from_numpy(np.array([sdf])).float(),
            'idx': mesh_idx
        }

        return sample

from model.SDF import *
from torch.utils import data

def normalize_mesh(mesh):
    """
    Get a trimesh and normalize it to the unit sphere
    This function works in place
    """
    mesh.vertices -= mesh.center_mass
    mesh.vertices /= np.linalg.norm(np.max(mesh.vertices))

def mesh_gen_df(mesh_file):
    mesh = trimesh.load(mesh_file)
    sample_nums = 50000
    sample_alpha = 0.8
    sample_alpha_normal = 0.2
    # 归一化
    mesh.vertices -= mesh.center_mass
    mesh.vertices /= np.linalg.norm(np.max(mesh.vertices))

    # 获取面上采样点
    sampled_pts, _ = trimesh.sample.sample_surface(mesh, int(sample_alpha * sample_nums), face_weight=None)
    
    # 添加噪声
    for ii in range(sampled_pts.shape[1]):
        sampled_pts[:, ii] += np.random.normal(0, 5e-2, int(0.8 * 50000))
    
    # 获取随机采样点
    samples_normal = np.random.normal(0, 0.7, (int((sample_alpha_normal) * sample_nums), 3))
    # 合并采样点
    samples = np.concatenate([sampled_pts, samples_normal])

    # 计算sdf
    pq = trimesh.proximity.ProximityQuery(mesh)
    sdf = pq.signed_distance(samples)

    # 生成dataframe
    df = pd.DataFrame({
        'x': samples[:, 0],
        'y': samples[:, 1],
        'z': samples[:, 2],
        'sdf': sdf
    })
    return df

def mesh_to_excel(in_dir, out_dir):
    meshes = os.listdir(in_dir)
    for mesh_name in meshes:
        mesh_name_pre, _ = mesh_name.split('.')
        print(f'gen file :{mesh_name_pre}.xlsx')
        if os.path.isfile(os.path.join(out_dir, mesh_name_pre + '.xlsx')):
            continue
        df = mesh_gen_df(os.path.join(in_dir, mesh_name))
        print(f'gen file :{mesh_name_pre}.xlsx success')
        print(len(df))
        df.to_excel(os.path.join(out_dir, mesh_name_pre + '.xlsx'))

if __name__ == '__main__':
    
    mesh_to_excel('./Thingi10K/raw_meshes/', 'stl_out')

    # 数据类测试代码    
    # dataset = MyDataset('./stl_out_1', 'cuda')
    # print(dataset.get_files_len())
    # print(dataset.__len__())
