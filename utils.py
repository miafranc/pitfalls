import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
import json
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import settings


def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tb_writer(logdir_prefix, filename):
    dirname = filename + '_' + datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
    logdir = os.path.join(logdir_prefix, dirname)
    return SummaryWriter(log_dir=logdir), dirname


def get_train_validation(dataset, use_percent=1, val_split=0.20, stratify=True):
    """
    Split data into train and validation sets.
    Added use_percent parameter.
    """
    labels = None
    if stratify:
        if isinstance(dataset, torch.utils.data.Subset):
            labels = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            labels = np.array(dataset.targets)

    if use_percent == 1:
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, stratify=labels)
    else:
        idx, _              = train_test_split(list(range(len(dataset))), test_size=1-use_percent, stratify=labels)
        train_idx, val_idx  = train_test_split(idx, test_size=val_split, stratify=labels[idx])
    
    return (torch.utils.data.Subset(dataset, train_idx), 
            torch.utils.data.Subset(dataset, val_idx))


def dump_parameters(logdir, filename):
    params = {}
    for p in dir(settings):
        if not p.startswith('_'):
            params[p] = getattr(settings, p)

    f = open(os.path.join(logdir, filename), 'w')
    json.dump(params, f, indent=2)
    f.close()


class ImageDatasetWithFilenames(Dataset):
    '''A custom image dataset is needed in order to know the filenames.
    Filenames are used to be able to store additional data of training images,
    e.g. Grad-CAM activations.
    '''
    
    def __init__(self, img_dir, transform=None, target_transform=None, 
                   repeat: int = 1):
        if repeat < 1:
            raise ValueError("`repeat` >= 1")
        
        self.repeat = repeat  
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        subfolders = sorted(os.listdir(self.img_dir))
        self.map = {subfolders[i]:i for i in range(len(subfolders))}
        self.imap = {i:subfolders[i] for i in range(len(subfolders))}
        self.images = []
        for sf in subfolders:
            for f in sorted(os.listdir(os.path.join(self.img_dir, sf))):
                self.images.append((f, self.map[sf]))

    
    def __len__(self):
        return len(self.images)*self.repeat

    
    def __getitem__(self, idx):
        real_idx = idx % len(self.images)   
        img_path = os.path.join(self.img_dir, self.imap[self.images[real_idx][1]], self.images[real_idx][0])
        
        image = Image.open(img_path)#.convert('RGB') # is convert needed?

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # image = read_image(img_path)
        label = self.images[real_idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label, self.imap[self.images[real_idx][1]], self.images[real_idx][0]
