import configargparse

def config_parser():
    parser = configargparse.ArgumentParser(description='Parathdetect')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='path to latest checkpoint')

    parser.add_argument('--start_epoch', type=int, default=0,
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--epochs', type=int, default=60,
                        help='total number of training rounds')

    parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')

    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--nBottleneck', type=int, default=4000, help='of dim for bottleneck of encoder')
    parser.add_argument('--overlapPred', type=int, default=4, help='overlapping edges')
    parser.add_argument('--nef', type=int, default=64, help='of encoder filters in first conv layer')
    parser.add_argument('--wtl2', type=float, default=0.998, help='0 means do not use else use with this weight')

    return parser

def get_config():
    parser = config_parser()
    cfg = parser.parse_args()
    return cfg

