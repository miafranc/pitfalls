'''
Based on: https://github.com/cfchen-duke/ProtoPNet/blob/master/img_aug.py
Modifications:
    - Added random seed;
    - Made the processing single-threaded; multi-threadding makes the process irreproducible;
    - Added try-except clauses to catch exceptions that sometimes occur when cropping the images. 
    For this, the p.process() had to be replaced by its definition from the code.
'''

import Augmentor
import os
from tqdm import tqdm

from utils import set_seed

SEED = 42


def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


datasets_root_dir = '/data/stanford_cars' # absolute path needed! # <- to be set
dir = os.path.join(datasets_root_dir, 'train_crop') # input dir # <- to be set
target_dir = os.path.join(datasets_root_dir, 'train_cropped_augmented') # output dir # <- to be set

makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

folders = sorted(folders)
target_folders = sorted(target_folders)
set_seed(SEED)

operations = ['rotate', 'skew', 'shear']

for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]

    for op in operations:
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        match op:
            case 'rotate':
                p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
            case 'skew':
                p.skew(probability=1, magnitude=0.2)
            case 'shear':
                p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        for i in range(10):
            with tqdm(total=len(p.augmentor_images), desc='Executing Pipeline', unit=' Samples') as progress_bar:
                for augmentor_image in p.augmentor_images:
                    try:
                        p._execute(augmentor_image)
                        progress_bar.set_description('Processing %s' % os.path.basename(augmentor_image.image_path))
                        progress_bar.update(1)
                    except Exception as e:
                        print(f'Error ({op}): {e}, folder: {fd}')
        del p
