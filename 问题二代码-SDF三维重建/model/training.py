from __future__ import division
import torch
import numpy as np
from torch.utils import data
import time
from datetime import datetime
class Trainer(object):
    def __init__(self, model, embedding, TrainDataset, args):
        self.args = args
        self.model = model
        self.embedding = embedding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.to(self.device)
            self.embedding.to(self.device)

        self.Trainloader = data.DataLoader(dataset=TrainDataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(lr=self.args.lr, params=model.parameters())
        self.optimizer_embedding = torch.optim.Adam(self.embedding.parameters(), lr=self.args.lr_embedding)
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch

    def train_model(self):
        self.model.train()
        self.embedding.train()
        epoch_start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs):

            epoch_loss = 0
            batch_loss = 0
            num_batches = len(self.Trainloader)

            for i, data in enumerate(self.Trainloader):
                xyz = data['xyz'].to(self.device)
                latent_idx = data['idx'].to(self.device)
                sdf = data['sdf'].to(self.device)

                latent = self.embedding(latent_idx)     # 通过形状序号生成潜向量
                self.model.zero_grad()
                self.embedding.zero_grad()
                output = self.model(xyz, latent)
                loss = self.criterion(
                    torch.clamp(output, -self.args.delta, self.args.delta),
                    torch.clamp(sdf, -self.args.delta, self.args.delta)
                )
                # 点云模型的loss
                # loss = self.criterion(output, sdf)
                loss.backward()
                self.optimizer.step()
                self.optimizer_embedding.step()
                epoch_loss += loss.item()
                batch_loss += loss.item()
                epoch_end_time = time.time()
                if (i + 1) % 100 == 0:  # 100batch 打印一次
                    print(f"Epoch [{epoch + 1}/{self.epochs}],  Batch [{i + 1}/{num_batches}], Loss: {batch_loss / 100:.4f}, Train Time: {epoch_end_time - epoch_start_time:.2f}s, Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    batch_loss = 0

            # 保存模型
            torch.save(self.model.state_dict(),
                       'SDF_checkpoint_e%d.pth' % epoch)
            torch.save(self.embedding.state_dict(),
                       'Emb_checkpoint_e%d.pth' % epoch)
