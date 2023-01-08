import argparse
import datetime
import os
import torch
from torch.utils.data import DataLoader
import yaml
from models.fcn32s import FCN32s
from models.fcn16s import FCN16s
from datasets.voc import SBDClassSeg, VOC2011ClassSeg 
from train_fcn32s import get_parameters
from trainer import Trainer

here = os.getcwd()

def train():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument('--resume', help='checkpoint path')
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    parser.add_argument('--max-iteration', type=int, default=100000, help='max iteration')
    parser.add_argument('--lr', type=float, default=1.0e-12, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.99, help='momentum')
    parser.add_argument('--pretrained-model', default='fcn32s.pth.tar', help='pretrained model of FCN32s')
    args = parser.parse_args()

    args.model = 'FCN16s'

    now = datetime.datetime.now()
    args.out = os.path.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(os.path.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # dataset
    root = os.path.expanduser('~/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_dl = DataLoader(
        SBDClassSeg(root, split='train', is_transform=True),
        batch_size=1, shuffle=True, **kwargs
    )
    val_dl = DataLoader(
        VOC2011ClassSeg(root, split='seg11valid', is_transform=True),
        batch_size=1, shuffle=False, **kwargs
    )

    # model

    model = FCN16s(n_class=21)
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        fcn32s = FCN32s(n_class=21)
        state_dict = torch.load(args.pretrained_model)
        try:
            fcn32s.load_state_dict(state_dict)
        except RuntimeError:
            fcn32s.load_state_dict(state_dict['model_state_dict'])
        model.copy_params_from_fcn32s(fcn32s)
    
    if cuda:
        model = model.cuda()

    # optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
            'lr': args.lr * 2, 'weight_decay': 0}
        ],
        lr = args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_dl=train_dl,
        val_dl=val_dl,
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=4000,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    train()