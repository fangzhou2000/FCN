import argparse
import datetime
import os
import torch
from torch.utils.data import DataLoader
import yaml
from models.fcn32s import FCN32s
from models.vgg import VGG16
from datasets.voc import SBDClassSeg, VOC2011ClassSeg 
from trainer import Trainer

def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        FCN32s
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


here = os.getcwd()

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
    args.out = os.path.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(os.path.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DIVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
    
    # dataset
    
    root = os.path.expanduser("~/datasets")
    kwargs = {"num_workers": 4, "pin_memory": True} if cuda else {}
    # the batch size must be 1 for SDS, because the size for images are not the same
    # if want mini-batch, need to resize the images and process the labels
    train_dl = DataLoader(
        SBDClassSeg(root=root, split='train', is_transform=True),
        batch_size=1, shuffle=True, **kwargs
    )
    val_dl = DataLoader(
        VOC2011ClassSeg(root=root, split='seg11valid', is_transform=True),
        batch_size=1, shuffle=False, **kwargs
    )
    
    # model
    model = FCN32s(n_class=21)
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    # optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True), 'lr': args.lr * 2, 'weight_decay': 0.0005}
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

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    train()


    

