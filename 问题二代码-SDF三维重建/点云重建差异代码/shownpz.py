import numpy as np
import open3d as o3d
# 加载 .npz 文件
data = np.load("output1.npz")
#data = np.load("../dataset/SDFdata_train/train/3c9d577c78dcc3904c3a35cee92bb95b.npz")

print(data.files)  # 查看文件中保存的数组名称
pos_points = data['pos']  # 包含 xyz 和 sdf 值
neg_points = data['neg']
print(pos_points.shape, neg_points.shape)
pos_xyz = pos_points[:, :3]
pos_sdf = pos_points[:, 3]
neg_xyz = neg_points[:, :3]
neg_sdf = neg_points[:, 3]

def save_to_ply(filename, xyz, sdf):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    # 可视化 SDF 值为颜色
    colors = np.zeros((xyz.shape[0], 3))
    colors[:, 0] = (sdf > 0)  # 正 sdf 用一种颜色
    colors[:, 2] = (sdf < 0)  # 负 sdf 用另一种颜色
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, point_cloud)

save_to_ply("pos_points.ply", pos_xyz, pos_sdf)
save_to_ply("neg_points.ply", neg_xyz, neg_sdf)

neg_points = o3d.io.read_point_cloud("neg_points.ply")

# 显示保存的neg形状
o3d.visualization.draw_geometries([neg_points])

