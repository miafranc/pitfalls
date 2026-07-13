import os
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import json
from PIL import Image
import numpy as np
from typing import Any, Union
import sys

import settings
from protopnet.preprocess import mean, std, preprocess_input_function
from utils import ImageDatasetWithFilenames, set_seed
from protopnet.model import PPNet
import protopnet.train_and_test


sys.path.insert(0, './protopnet')


class PrototypePurity:
    '''Prototype purity (for the CUB dataset).
    Not necessarily gives the exact same score as the implementation in https://github.com/M-Nauta/PIPNet/,
    because there is no filtering out prototypes having small classification weights.
    '''

    def __init__(self, dataset: str = 'test', device: str = 'cuda:0') -> None:
        data_path = settings.TEST_DIR[:settings.TEST_DIR.find('/test')]
        self.data_path = data_path
        # print(data_path)
        # exit(0)

        self.images_file = os.path.join(data_path, 'images.txt')
        self.parts_file = os.path.join(data_path, 'parts.txt')
        self.part_locs_file = os.path.join(data_path, 'part_locs.txt')

        self.device = device

        # Reading images.txt:
        f = open(self.images_file, 'r')
        data = f.readlines()
        f.close()
        self.image_map = {} # filename: ID
        self.image_map_r = {} # ID: filename
        for r in data:
            rr = r.strip().split(' ')
            filename_only = rr[1][rr[1].find('/') + 1:] # remove folder name
            self.image_map[filename_only] = (int(rr[0]), rr[1])
            self.image_map_r[int(rr[0])] = filename_only

        # Reading part_locs.txt:
        self.parts = {} # filename: [(part_type_1, x_1, y_1), (part_type_2, x_2, y_2), ...]
        f = open(self.part_locs_file, 'r')
        data = f.readlines()
        f.close()
        for r in data:
            rr = r.strip().split(' ')
            fid, part, x, y, present = int(rr[0]), int(rr[1]), float(rr[2]), float(rr[3]), int(rr[4])
            if present:
                if self.parts.get(fid, -1) == -1:
                    self.parts[fid] = [(part, x, y)]
                else:
                    self.parts[fid].append((part, x, y))

        # Reading parts.txt:
        self.parts_map = {}
        f = open(self.parts_file, 'r')
        data = f.readlines()
        f.close()
        for r in data:
            rr = r.strip().split(' ')
            part_id, part_name = int(rr[0]), str.lower(' '.join(rr[1:]))
            self.parts_map[part_id] = part_name
        # Part map: left part id -> right part id
        part_map_r = {v:k for k, v in self.parts_map.items()}
        self.duplicate_parts = {}
        for id, part in self.parts_map.items():
            if part.startswith('left'):
                right_name = part.replace('left', 'right')
                self.duplicate_parts[id] = part_map_r[right_name]

        # We need the uncropped dataset:
        assert dataset in ['test', 'train']
        self.dataset = dataset
        test_root = os.path.join(data_path, dataset)
        test_tf = v2.Compose([
            v2.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std)
        ])
        testset = ImageDatasetWithFilenames(test_root, transform=test_tf)
        self.testloader = DataLoader(
            testset,
            batch_size=settings.BATCH_SIZE,
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )

        self.active_patches_filename = os.path.join(data_path, 'active_patches.json')
        self.feature_map_size = [1, 1] # just init


    def _get_img_patch_coordinates(self, linear_index: int) -> tuple[int, int, int, int]:
        h_min = h_max = w_min = w_max = 0

        # Assume image height and width are the same
        hindex = linear_index // self.feature_map_size[0]
        windex = linear_index % self.feature_map_size[0]
        patchsize = settings.IMG_SIZE // self.feature_map_size[0]
        h_min = hindex * patchsize
        h_max = (hindex + 1) * patchsize - 1
        w_min = windex * patchsize
        w_max = (windex + 1) * patchsize - 1
        
        return (h_min, h_max, w_min, w_max)


    def _get_scaled_coordinate(self, h: int, w: int, filename: str) -> tuple[int, int]:
        img = Image.open(os.path.join(self.data_path, self.dataset, filename))
        img_width, img_height = img.size
        return ((h * settings.IMG_SIZE) // img_height, (w * settings.IMG_SIZE) // img_width)


    def purity(self,
               model: PPNet, 
               overwrite: bool = False, 
               top_k: int = 10, 
               use_sigmoid = False,
               threshold_activation: float = 0.0,
               duplicates: bool = False,
               ignore_small_weights: bool = False,
               smooth: bool = False,
               gamma: float = 1.0,
               margin: float = 0.0,
               prototype_activation_function: str = 'log') -> list[Any]:
        
        active_patches = {} # structure: {proto_index: [filename, max_activation_score, linear_index_of_max_activated_patch], ...]
        topk_patches = {}

        # Get size of feature map
        self.feature_map_size = model.conv_features(torch.rand(1, 3, settings.IMG_SIZE, settings.IMG_SIZE).to(self.device)).shape[2:]

        classification_weights = torch.max(model.last_layer.weight, dim=0)[0]

        if not os.path.exists(self.active_patches_filename) or overwrite:
            # NOTE: We do not filter out prototypes based on the weights, as in PIP-Net. 
            # In PIP-Net, if the max. class. weight of a prototype was below a threshold, the corresponding prototype was not taken into account.
            with tqdm(self.testloader, unit='batch', desc='Finding most active patches: ') as t:
                for i, data in enumerate(t):
                    (image, _, _, filename) = data

                    input = image.to(self.device)

                    with torch.no_grad():
                        output, _ = model(input)
                        prototype_activation_patterns = model.prototype_distances(input)
                        if prototype_activation_function == 'log':
                            prototype_activation_patterns = model.distance_2_similarity(prototype_activation_patterns)
                        else:
                            prototype_activation_patterns *= -1
                        if use_sigmoid:
                            prototype_activation_patterns = F.sigmoid(prototype_activation_patterns)

                    pooled_activations = torch.amax(prototype_activation_patterns, dim=(2, 3)) # max. activation score for every prototype
                    
                    # Purity from PIP-Net ignored prototypes not relevant to any classes:
                    if ignore_small_weights:
                        small_weights_ind = torch.where(classification_weights < 1e-5)[0]
                        if len(small_weights_ind) > 0:
                            pooled_activations[:, small_weights_ind] = -1

                    for batch_i in range(output.shape[0]): # for every image in the batch
                        ind_active = torch.where(pooled_activations[batch_i] > threshold_activation)
                        proto_active_activation_patterns = prototype_activation_patterns[batch_i, ind_active[0]]
                        max_activations, max_ind = torch.max(torch.flatten(proto_active_activation_patterns, 1, 2), dim=1) # linearized indices (flatten)

                        for ind, p_i in enumerate(ind_active[0]):
                            if active_patches.get(p_i.item(), -1) == -1:
                                active_patches[p_i.item()] = [[filename[batch_i], max_activations[ind].item(), max_ind[ind].item()]]
                            else:
                                active_patches[p_i.item()].append([filename[batch_i], max_activations[ind].item(), max_ind[ind].item()])

            topk_patches = {}
            for p_i in tqdm(active_patches.keys(), desc=f'Finding top-{top_k} patches: '):
                if top_k > 0:
                    topk_patches[p_i] = sorted(active_patches[p_i], key=lambda x: x[1], reverse=True)[:top_k]
                else:
                    topk_patches[p_i] = sorted(active_patches[p_i], key=lambda x: x[1], reverse=True)[:]
            
            f = open(self.active_patches_filename, 'w')
            json.dump(topk_patches, f, indent=2)
            f.close()
        
        else: 
            print('Loading top-k active patches from file.')
            f = open(self.active_patches_filename, 'r')
            topk_patches = json.load(f)
            f.close()

        proto_purity = {}
        for p_i in tqdm(topk_patches.keys(), desc='Calculating prototype purity: '): # prototype id
            for filename, _, max_ind in topk_patches[p_i]:
                fileid = self.image_map[filename][0]
                h_min, h_max, w_min, w_max = self._get_img_patch_coordinates(max_ind)
                for part in self.parts[fileid]:
                    # First is width, second is height coordinate in part_locs.txt:
                    h_feature, w_feature = self._get_scaled_coordinate(part[2], part[1], self.image_map[filename][1])

                    part_id = part[0]

                    part_in_patch = 0
                    if h_feature >= h_min and h_feature <= h_max and w_feature >= w_min and w_feature <= w_max:
                        part_in_patch = 1
                    else:
                        if smooth:
                            d_min_h = 0 if h_feature >= h_min and h_feature <= h_max else min(abs(h_feature - h_min), abs(h_feature - h_max))
                            d_min_w = 0 if w_feature >= w_min and w_feature <= w_max else min(abs(w_feature - w_min), abs(w_feature - w_max))
                            d = np.sqrt(d_min_h**2 + d_min_w**2)
                            part_in_patch = np.exp(-gamma * max(d - margin, 0)**2)

                    if proto_purity.get(p_i, -1) == -1:
                        proto_purity[p_i] = {part_id: [part_in_patch]}
                    else:
                        if proto_purity[p_i].get(part_id, -1) == -1:
                            proto_purity[p_i][part_id] = [part_in_patch]
                        else:
                            proto_purity[p_i][part_id].append(part_in_patch)
                if duplicates:
                    # "Remove" duplicate parts:
                    img_parts = {pid: (px, py) for pid, px, py in self.parts[fileid]}
                    for part_left, part_right in self.duplicate_parts.items():
                        if part_left in img_parts.keys():
                            if part_right in img_parts.keys():
                                # If left and right part present in the same image:
                                presence_left = proto_purity[p_i][part_left][-1]
                                presence_right = proto_purity[p_i][part_right][-1]
                                if presence_left > presence_right:
                                    proto_purity[p_i][part_right][-1] = presence_left
                            else:
                                # If left part is present but not the right part, leave only the right part:
                                if part_right not in proto_purity[p_i].keys():
                                    proto_purity[p_i][part_right] = []
                                proto_purity[p_i][part_right].append(proto_purity[p_i][part_left][-1])
                            del proto_purity[p_i][part_left]

        proto_purity_score = {}
        for p_i in proto_purity.keys():
            proto_purity_score[p_i] = np.max([np.mean(v) for v in proto_purity[p_i].values()])

        return [v for v in proto_purity_score.values()]


def calculate_purity(model_file):
    p = PrototypePurity(dataset='test', device=settings.DEVICE)
    ppnet = torch.load(model_file, weights_only=False)
    ppnet = ppnet.to(settings.DEVICE)
    ppnet.eval()
    p.active_patches_filename = 'logs/active_patches_' + os.path.basename(model_file) + '.json'
    proto_purity = p.purity(ppnet,
                            overwrite=True, 
                            top_k=10,
                            use_sigmoid=False,
                            threshold_activation=-np.inf,
                            duplicates=True,
                            ignore_small_weights=False,
                            smooth=False,
                            prototype_activation_function='linear')
    print(f'{"Prototype purity":>42}: {np.mean(proto_purity):.6f} +/- {np.std(proto_purity):.6f}')


def train_test_protopnet(model_file):
    normalize = transforms.Normalize(mean=mean, std=std)
    test_dataset = datasets.ImageFolder(
        settings.TEST_DIR,
        transforms.Compose([
            transforms.Resize(size=(settings.IMG_SIZE, settings.IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=settings.BATCH_SIZE, 
                                              shuffle=False,
                                              num_workers=4, 
                                              pin_memory=False)
 
    ppnet = torch.load(model_file, weights_only=False)
    ppnet = ppnet.to(settings.DEVICE)
    ppnet.eval()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    accu = protopnet.train_and_test.test(model=ppnet_multi, 
                                         dataloader=test_loader,
                                         class_specific=True, 
                                         log=print)


if __name__ == '__main__':
    settings.DEVICE = 'cuda:0'
    torch.cuda.set_device(settings.DEVICE)
    print(torch.cuda.get_device_name(settings.DEVICE))
    
    settings.SEED = 123
    set_seed(settings.SEED)

    model_file = 'models/protopnet/123/20_5push0.7768.pth'

    # calculate_purity(model_file)

    train_test_protopnet(model_file)
    