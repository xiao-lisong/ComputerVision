import configs.config_loader as cfg_loader
from model.GNet import *
from model.DNet import *
from model.training import *
from dataset import *


def main():
    args = cfg_loader.get_config()
    gnet = GNet(args)
    dnet = DNet(args)
    # 用于读取已训练的模型继续训练
    # gnet_checkpoint = torch.load("GNetDict.pth")
    # dnet_checkpoint = torch.load("DNetDict.pth")

    # gnet.load_state_dict(gnet_checkpoint['state_dict'])
    # dnet.load_state_dict(dnet_checkpoint['state_dict'])
    # start_epoch = gnet_checkpoint['epoch']
    # print(gnet)
    # print(dnet)
    # print(start_epoch)
    # args.start_epoch = start_epoch
    
    TrainDataset = MyDataset(args.imageSize, './data/train')
    trainer = Trainer(gnet, dnet, TrainDataset, args)
    trainer.train_model()


if __name__ == '__main__':
    main()
