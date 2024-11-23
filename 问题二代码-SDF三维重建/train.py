import configs.config_loader as cfg_loader
from model.SDF import *
from model.training import *
from dataset import *

def main():
    args = cfg_loader.get_config()
    model = SDF(args.latent_dim, args.hidden_dim)
    TrainDataset = MyDataset('./data', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    embedding = LatentEmbedding(TrainDataset.get_files_len(), args.latent_dim)
    # 加载已经训练的模型继续训练
    # model.load_state_dict(torch.load('SDF_checkpoint_e10.pth', weights_only=True))
    # embedding.load_state_dict(torch.load('Emb_checkpoint_e10.pth', weights_only=True))
    # args.start_epoch = 10
    trainer = Trainer(model, embedding, TrainDataset, args)
    trainer.train_model()

if __name__ == '__main__':
    main()
