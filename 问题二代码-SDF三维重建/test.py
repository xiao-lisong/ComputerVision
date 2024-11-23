from dataset import *
from torchvision import transforms
from model.SDF import SDF
import numpy as np
import math
import tqdm
import mcubes

device = torch.device("cuda")

def test(input, output, epoch):
    model = SDF(latent_dim=256, hidden_dim=512)
    # 加载训练的模型
    sdf_path = f'SDF_checkpoint_e{epoch}.pth'
    emb_path = f'Emb_checkpoint_e{epoch}.pth'
    model.load_state_dict(torch.load(sdf_path, weights_only=True))
    model.eval()
    model.to(device)
    embedding = LatentEmbedding(296, 256) 
    embedding.load_state_dict(torch.load(emb_path, weights_only=True))
    embedding.eval()
    embedding.to(device)
    criterion = torch.nn.MSELoss()

    # 通过三维坐标计算SDF
    eps=0.005
    latent_z = embedding(torch.tensor(input).to(device)).detach().cpu()
    with torch.no_grad():
        n = math.ceil(int(2.0 / eps))
        grid = np.zeros((n, n, n))
        latent = np.zeros((n, 256))
        for i in range(n):
            latent[i, :] = latent_z
        latent = torch.from_numpy(latent).float().to(device)
    for i, x in tqdm.tqdm(enumerate(np.arange(-1.0, 1.0, eps)), total=n):
        for j, y in enumerate(np.arange(-1.0, 1.0, eps)):
            batch = np.zeros((n, 3))
            for k, z in enumerate(np.arange(-1.0, 1.0, eps)):
                batch[k, 0] = x
                batch[k, 1] = y
                batch[k, 2] = z
            xyz = torch.from_numpy(batch).float().to(device)
            with torch.no_grad():
                val = model(xyz, latent)
            grid[i, j, :] = val.cpu().numpy().reshape(-1)
    vertices, triangles = mcubes.marching_cubes(grid, 0)

    # 保存结果到stl文件
    mesh = trimesh.Trimesh(vertices, triangles)
    normalize_mesh(mesh)
    mesh.export(output)

if __name__ == '__main__':
    input = 10
    output = "output5.stl"
    test(input, output, 190)