from dataset import *
from torchvision import transforms
from model.SDF import SDF
import numpy as np
device = torch.device("cpu")


# 生成均匀分布在单位球内的点
def random_points_in_unit_sphere(num_points):
    points = []
    while len(points) < num_points:
        # 随机生成一个 [-1, 1]^3 的点
        point = np.random.uniform(-1, 1, 3)
        # 保证点在单位球内
        if np.linalg.norm(point) <= 1:
            points.append(point)
    return np.array(points)


def test(input, output, epoch):
    model = SDF(latent_dim=256, hidden_dim=512)
    sdf_path = f'pth_point/SDF_checkpoint_e{epoch}.pth'
    emb_path = f'pth_point/Emb_checkpoint_e{epoch}.pth'
    model.load_state_dict(torch.load(sdf_path, weights_only=True))
    model.eval()

    embedding = LatentEmbedding(25, 256) 
    embedding.load_state_dict(torch.load(emb_path, weights_only=True))
    embedding.eval()
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        latent = embedding(torch.tensor(input))
        # 生成3D随机点
        xyz = random_points_in_unit_sphere(5000000)
        xyz = torch.from_numpy(xyz).float()
        latent = latent.repeat(xyz.size(0), 1)
        output = model(xyz, latent)
        print(output)

        seg = 0.0      # 分割阈值
        pos_mask = output > seg
        neg_mask = output < seg
        pos_xyz = xyz[pos_mask]
        neg_xyz = xyz[neg_mask]

        pos_sdf = output[pos_mask]
        neg_sdf = output[neg_mask]
        pos_data = np.concatenate((pos_xyz, pos_sdf[:, np.newaxis]), axis=1)
        neg_data = np.concatenate((neg_xyz, neg_sdf[:, np.newaxis]), axis=1)
        print(pos_data.shape)
        print(neg_data.shape)
        npz_output = {}
        npz_output['pos'] = neg_data
        npz_output['neg'] = neg_data
        
        np.savez("output1.npz", **npz_output)

if __name__ == '__main__':
        
    input = 10
    output = "output1.npz"
    test(input, output, 92)