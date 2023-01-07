import os
import collections
import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils.data import Dataset

class VOCClassSegBase(Dataset):
    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor'
    ])

    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split='train', is_transform=False):
        self.root = root
        self.split = split
        self.is_transform = is_transform

        # VOC2011 is subset of VOC2012
        dataset_dir = os.path.join(root, 'VOC2012/VOCdevkit/VOC2012')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imagesets_file = os.path.join(dataset_dir, 'ImageSets/Segmentation/{}.txt'.format(split))
            for id in open(imagesets_file):
                id = id.strip()
                image_file = os.path.join(dataset_dir, 'JPEGImages/{}.jpg'.format(id))
                label_file = os.path.join(dataset_dir, 'SegmentationClass/{}.png'.format(id))
                self.files[split].append({
                    'image': image_file,
                    'label': label_file
                })
            
    def __len__(self):
        return len(self.files[self.split])
    
    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        image_file = data_file['image']
        image = PIL.Image.open(image_file) # HWC, RGB
        image = np.array(image, dtype=np.uint8)
        # load label
        label_file = data_file['label']
        label = PIL.Image.open(label_file)
        label = np.array(label, dtype=np.int32)
        label[label == 255] = -1
        if self.is_transform:
            return self.transform(image, label)
        else:
            return image, label

    def transform(self, image, label):
        image = image[:, :, ::-1] # RGB -> BGR
        image = image.astype(np.float64)
        image -= self.mean_bgr
        image = image.transpose(2, 0, 1) # HWC -> CHW
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        return image, label
    
    def untransform(self, image, label):
        image = image.numpy()
        image = image.transpose(1, 2, 0)
        image += self.mean_bgr
        image = image.astype(np.uint8)
        label = label.numpy()
        return image, label
    
class VOC2011ClassSeg(VOCClassSegBase):

    def __init__(self, root, split='train', is_transform=False):
        super(VOC2011ClassSeg, self).__init__(root=root, split=split, is_transform=is_transform)
        # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/seg11valid.txt
        imagesets_file = os.path.join(self.root, 'VOC2012/seg11valid.txt')
        dataset_dir = os.path.join(self.root, 'VOC2012/VOCdevkit/VOC2012')
        for id in open(imagesets_file):
            id = id.strip()
            image_file = os.path.join(dataset_dir, 'JPEGImages/{}.jpg'.format(id))
            label_file = os.path.join(dataset_dir, 'SegmentationClass/{}.png'.format(id))
            self.files['seg11valid'].append({
                'image': image_file,
                'label': label_file
            })

class VOC2012ClassSeg(VOCClassSegBase):

    def __init__(self, root, split='train', is_transform=False):
        super(VOC2012ClassSeg, self).__init__(root=root, split=split, is_transform=is_transform)
    
class SBDClassSeg(VOCClassSegBase):

    def __init__(self, root, split='train', is_transform=False):
        self.root = root
        self.split = split
        self.is_transform = is_transform

        dataset_dir = os.path.join(self.root, 'VOC2012/benchmark_RELEASE/dataset')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imagesets_file = os.path.join(dataset_dir, '{}.txt'.format(split))
            for id in open(imagesets_file):
                id = id.strip()
                image_file = os.path.join(dataset_dir, 'img/{}.jpg'.format(id))
                label_file = os.path.join(dataset_dir, 'cls/{}.mat'.format(id))
                self.files[split].append({
                    'image': image_file,
                    'label': label_file
                })
    
    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        image_file = data_file['image']
        image = PIL.Image.open(image_file)
        image = np.array(image, dtype=np.uint8)
        # load label
        label_file = data_file['label']
        mat = scipy.io.loadmat(label_file)
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
        label[label ==255] = -1
        if self.is_transform:
            return self.transform(image, label)
        else:
            return image, label