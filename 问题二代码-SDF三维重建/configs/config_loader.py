import configargparse

def config_parser():
    parser = configargparse.ArgumentParser(description='Parathdetect')

    parser.add_argument('--batch_size', type=int, default=2048,
                        help='path to latest checkpoint')

    parser.add_argument('--start_epoch', type=int, default=0,
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--epochs', type=int, default=60,
                        help='total number of training rounds')

    parser.add_argument('--latent_dim', type=int, default=256, help='latent_dim')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--lr', type=float, default=1e-5, help='lr')
    parser.add_argument('--lr_embedding', type=float, default=1e-3, help='lr')
    parser.add_argument('--delta', type=float, default=0.05, help='lr')

    parser.add_argument('-f', type=None, default=0,
                        help='jupyter default')

    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()
    return cfg

