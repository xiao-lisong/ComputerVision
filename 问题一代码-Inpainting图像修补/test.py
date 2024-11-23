from dataset import *
from torchvision import transforms
from torch.utils import data
from torch.autograd import Variable
import torchvision.utils as vutils
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imageSize = 128
overlapPred = 4
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
batch_size=48
# 进行反归一化 使保存的图像看起来正常
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    # 反归一化操作
    tensor = tensor * std + mean
    return tensor

def test():
    os.makedirs("result/test", exist_ok=True)

    #加载模型
    model = torch.load('GNet.pth', map_location=device)
    model.eval()    # 设置为评估模式
    #print(model)
    criterionMSE = torch.nn.MSELoss()
    # 测试集
    TestDataset = MyDataset(imageSize, './data/test')
    Testloader = data.DataLoader(dataset=TestDataset, batch_size=batch_size, shuffle=False, num_workers=0)

    input_cropped = torch.FloatTensor(batch_size, 3, imageSize, imageSize)
    input_cropped = Variable(input_cropped)

    real_center = torch.FloatTensor(batch_size, 3, int(imageSize / 2), int(imageSize / 2))

    if torch.cuda.is_available():
        input_cropped = input_cropped.cuda()
        model = model.cuda()
        real_center = real_center.cuda()

    # 获取模型输出结果
    for i, images in enumerate(Testloader):
        real_cpu = images.to(device)
        real_center_cpu = real_cpu[:, :, int(imageSize / 4):int(imageSize / 4) + int(imageSize / 2),
                                  int(imageSize / 4):int(imageSize / 4) + int(imageSize / 2)]
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)
        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)
        input_cropped.data[:, 0,
        int(imageSize / 4 + overlapPred):int(
            imageSize / 4 + imageSize / 2 - overlapPred),
        int(imageSize / 4 + overlapPred):int(
            imageSize / 4 + imageSize / 2 - overlapPred)] = 2 * 117.0 / 255.0 - 1.0
        input_cropped.data[:, 1,
        int(imageSize / 4 + overlapPred):int(
            imageSize / 4 + imageSize / 2 - overlapPred),
        int(imageSize / 4 + overlapPred):int(
            imageSize / 4 + imageSize / 2 - overlapPred)] = 2 * 104.0 / 255.0 - 1.0
        input_cropped.data[:, 2,
        int(imageSize / 4 + overlapPred):int(
            imageSize / 4 + imageSize / 2 - overlapPred),
        int(imageSize / 4 + overlapPred):int(
            imageSize / 4 + imageSize / 2 - overlapPred)] = 2 * 123.0 / 255.0 - 1.0
        print(real_cpu.shape)
        print(real_center_cpu.shape)
        fake = model(input_cropped)
        print(fake.shape)
        print(real_center.shape)
        loss = criterionMSE(fake, real_center)
 
        print(f'loss:{loss}')
        recon_image = input_cropped.clone()
        # 将生成的图像和原始图像拼接在一起
        recon_image.data[:, :, int(imageSize / 4):int(imageSize / 4 + imageSize / 2),
                    int(imageSize / 4):int(imageSize / 4 + imageSize / 2)] = fake.data
        recon_image = denormalize(recon_image, mean, std)
        # 保存图像
        grid = vutils.make_grid(recon_image[0:batch_size], nrow=8, padding=6, normalize=True)
        vutils.save_image(grid,
                                      'result/test/%02d.png'%(i + 1))
if __name__ == '__main__':
    test()