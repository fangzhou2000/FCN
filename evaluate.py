import os
import numpy as np
import argparse
import skimage.io
import tqdm
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import utils
from datasets.voc import VOC2011ClassSeg
from models.fcn32s import FCN32s


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model Path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    model_file = args.model_file

    root = os.path.expanduser('~/datasets')
    val_dl = DataLoader(
        VOC2011ClassSeg(root, split='seg11valid', is_transform=True),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    n_class = len(val_dl.dataset.class_names)

    model = FCN32s(n_class=n_class)

    if torch.cuda.is_available():
        model = model.cuda()

    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
        print('Loaded model_data')
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
        print("Loaded model_data['model_state_dict']")
    
    model.eval()
    
    print('==> Evaluating with VOC2011ClassSeg seg11valid')
    visualizations = []
    label_trues, label_preds = [], []
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(val_dl),
                                               total=len(val_dl),
                                               ncols=80, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        imgs = data.data.cpu()
        lbl_pred = output.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = val_dl.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) < 9:
                viz = utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
                    label_names=val_dl.dataset.class_names
                )
                visualizations.append(viz)
    metrics = utils.label_accuracy_score(
        label_trues, label_preds, n_class=n_class
    )
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))

    viz = utils.get_tile_image(visualizations)
    skimage.io.imsave('viz_evaluate.png', viz)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    eval()