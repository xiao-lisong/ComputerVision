import torch
from torch import nn

# 潜向量定义
class LatentEmbedding(nn.Module):
    def __init__(self, num_models, latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_models, latent_dim)
        torch.nn.init.uniform_(self.embedding.weight, 0, 0.1)

    def forward(self, x):
        return self.embedding(x)

# SDF网络模型定义
class SDF(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(SDF, self).__init__()
        self.fc1 = nn.utils.parametrizations.weight_norm(nn.Linear(latent_dim + 3, hidden_dim))
        self.fc2 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc4 = nn.utils.parametrizations.weight_norm(
            nn.Linear(hidden_dim, hidden_dim - latent_dim - 3))
        self.fc5 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc6 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc7 = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc8 = nn.Linear(hidden_dim, 1)

    def forward(self, xyz, latent):
        input = torch.cat((xyz, latent), dim=1)
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        # The skip connection described in the article
        x = torch.cat((x, input), dim=1)
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.tanh(self.fc8(x))

        # 训练点云模型时 fc8用以下的方式
        # x = self.fc8(x)
        # x = torch.squeeze(x, dim=-1)
        return x