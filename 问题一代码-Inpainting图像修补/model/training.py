from __future__ import division
import os
import torch
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
import time
from datetime import datetime
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

class Trainer(object):
    def __init__(self, gnet, dnet, TrainDataset, opt):
        self.opt = opt
        self.Gnet = gnet
        self.Dnet = dnet
        if torch.cuda.is_available():
            self.Gnet.cuda()
            self.Dnet.cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.Trainloader = data.DataLoader(dataset=TrainDataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
        self.criterionBCE = torch.nn.BCELoss()
        self.criterionMSE = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(lr=1e-3, params=self.Gnet.parameters())
        self.epochs = opt.epochs
        self.start_epoch = opt.start_epoch
        self.optimizerD = optim.Adam(self.Dnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.Gnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.wtl2 = float(opt.wtl2)

        if torch.cuda.is_available():
            self.Gnet.cuda()
            self.Dnet.cuda()
            self.criterionBCE.cuda()
            self.criterionMSE.cuda()

        os.makedirs("result/train/cropped", exist_ok=True)
        os.makedirs("result/train/real", exist_ok=True)
        os.makedirs("result/train/recon", exist_ok=True)

    # 进行反归一化 使保存的图像看起来正常
    def denormalize(self, tensor, mean, std):
        mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)

        # 反归一化操作
        tensor = tensor * std + mean
        return tensor

    def train_model(self):
        self.Gnet.train()
        self.Dnet.train()
        epoch_start_time = time.time()
        input_real = torch.FloatTensor(self.opt.batch_size, 3, self.opt.imageSize, self.opt.imageSize)
        input_cropped = torch.FloatTensor(self.opt.batch_size, 3, self.opt.imageSize, self.opt.imageSize)
        label = torch.FloatTensor(self.opt.batch_size)

        real_center = torch.FloatTensor(self.opt.batch_size, 3, int(self.opt.imageSize / 2), int(self.opt.imageSize / 2))

        real_label = 1
        fake_label = 0
        input_real = Variable(input_real)
        input_cropped = Variable(input_cropped)
        label = Variable(label)
        overlapL2Weight = 10

        if torch.cuda.is_available():
            print("using gpu")
            input_real, input_cropped, label = input_real.cuda(), input_cropped.cuda(), label.cuda()
            real_center = real_center.cuda()


        for epoch in range(self.start_epoch, self.epochs):
            epoch_loss = 0
            num_batches = len(self.Trainloader)
            for i, images in enumerate(self.Trainloader):
                real_cpu = images.to(self.device)
                real_center_cpu = real_cpu[:, :, int(self.opt.imageSize / 4):int(self.opt.imageSize / 4) + int(self.opt.imageSize / 2),
                                  int(self.opt.imageSize / 4):int(self.opt.imageSize / 4) + int(self.opt.imageSize / 2)]
                batch_size = real_cpu.size(0)
                input_real.data.resize_(real_cpu.size()).copy_(real_cpu)
                input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
                real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)
                input_cropped.data[:, 0,
                int(self.opt.imageSize / 4 + self.opt.overlapPred):int(self.opt.imageSize / 4 + self.opt.imageSize / 2 - self.opt.overlapPred),
                int(self.opt.imageSize / 4 + self.opt.overlapPred):int(
                    self.opt.imageSize / 4 + self.opt.imageSize / 2 - self.opt.overlapPred)] = 2 * 117.0 / 255.0 - 1.0
                input_cropped.data[:, 1,
                int(self.opt.imageSize / 4 + self.opt.overlapPred):int(self.opt.imageSize / 4 + self.opt.imageSize / 2 - self.opt.overlapPred),
                int(self.opt.imageSize / 4 + self.opt.overlapPred):int(
                    self.opt.imageSize / 4 + self.opt.imageSize / 2 - self.opt.overlapPred)] = 2 * 104.0 / 255.0 - 1.0
                input_cropped.data[:, 2,
                int(self.opt.imageSize / 4 + self.opt.overlapPred):int(self.opt.imageSize / 4 + self.opt.imageSize / 2 - self.opt.overlapPred),
                int(self.opt.imageSize / 4 + self.opt.overlapPred):int(
                    self.opt.imageSize / 4 + self.opt.imageSize / 2 - self.opt.overlapPred)] = 2 * 123.0 / 255.0 - 1.0

                # train with real
                self.Dnet.zero_grad()
                label.data.resize_(batch_size).fill_(real_label)

                output = self.Dnet(real_center)
                output = output.squeeze(1)
                errD_real = self.criterionBCE(output, label)
                errD_real.backward()
                D_x = output.data.mean()

                # train with fake
                # noise.data.resize_(batch_size, nz, 1, 1)
                # noise.data.normal_(0, 1)
                #print(input_cropped.shape)
                fake = self.Gnet(input_cropped)
                label.data.fill_(fake_label)
                output = self.Dnet(fake.detach()).squeeze(1)
                errD_fake = self.criterionBCE(output, label)
                errD_fake.backward()
                D_G_z1 = output.data.mean()
                errD = errD_real + errD_fake
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.Gnet.zero_grad()
                label.data.fill_(real_label)  # fake labels are real for generator cost
                output = self.Dnet(fake).squeeze(1)
                errG_D = self.criterionBCE(output, label)
                # errG_D.backward(retain_variables=True)

                #errG_l2 = self.criterionMSE(fake,real_center)
                wtl2Matrix = real_center.clone()
                wtl2Matrix.data.fill_(self.wtl2 * overlapL2Weight)
                wtl2Matrix.data[:, :, int(self.opt.overlapPred):int(self.opt.imageSize / 2 - self.opt.overlapPred),
                int(self.opt.overlapPred):int(self.opt.imageSize / 2 - self.opt.overlapPred)] = self.wtl2

                errG_l2 = (fake - real_center).pow(2)
                errG_l2 = errG_l2 * wtl2Matrix
                errG_l2 = errG_l2.mean()

                errG = (1 - self.wtl2) * errG_D + self.wtl2 * errG_l2
                errG.backward()
                D_G_z2 = output.data.mean()
                self.optimizerG.step()
                
                if i % 100 == 0:
                    epoch_end_time = time.time()
                    print(f"Epoch [{epoch + 1}/{self.epochs}],  Batch [{i + 1}/{num_batches}], Loss_D: {errD.item():.4f} "+
                      f"Loss_G: {errG_D.item():.4f} / {errG_l2.item():.4f} l_D(x): {D_x:.4f} l_D(G(z)): {D_G_z1:.4f} ', Train Time: {epoch_end_time - epoch_start_time:.2f}s, Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    vutils.save_image(self.denormalize(real_cpu, mean, std),
                                      'result/train/real/real_samples_epoch_%03d.png' % epoch)
                    vutils.save_image(self.denormalize(input_cropped.data, mean, std),
                                      'result/train/cropped/cropped_samples_epoch_%03d.png' % epoch)
                    recon_image = input_cropped.clone()
                    recon_image.data[:, :, int(self.opt.imageSize / 4):int(self.opt.imageSize / 4 + self.opt.imageSize / 2),
                    int(self.opt.imageSize / 4):int(self.opt.imageSize / 4 + self.opt.imageSize / 2)] = fake.data
                    vutils.save_image(self.denormalize(recon_image.data, mean, std),
                                      'result/train/recon/recon_center_samples_epoch_%03d.png' % epoch)
            # do checkpointing
            torch.save({'epoch':epoch+1,
                        'state_dict':self.Gnet.state_dict()},
                        'GNetDict.pth' )
            torch.save({'epoch':epoch+1,
                        'state_dict':self.Dnet.state_dict()},
                        'DNetDict.pth' )

        # 保存模型
        torch.save(self.Gnet, "GNet.pth")
        torch.save(self.Dnet, "DNet.pth")