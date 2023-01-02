import argparse
import datetime
import os
import torch
import yaml
from models import fcn32s

def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        fcn32s.FCN32s
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # freeze weight
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError("Unexpected module: {}".format(str(m)))


here = os.path.join(os.path.abspath(__file__))

def train():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--resume', help='checkpoint path')
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument('--max-iteration', type=int, default=100000, help='max iteration')
    parser.add_argument('--lr', type=float, default=1.0e-10, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.99, help='momentum')
    args = parser.parse_args()

    args.model = 'FCN32s'
    now = datetime.datetime.now()
    args.out = os.path.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%.f'))

    os.makedirs(args.out)
    with open(os.path.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DIVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
    
    # dataset
    
    

