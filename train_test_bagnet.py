import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pprint import pprint
import numpy as np
from torchvision.transforms import v2
import torchvision.datasets as datasets
import os
import argparse
import time
from PIL import Image
import matplotlib.pyplot as plt
from functools import reduce
import json
import re

import bagnet.bagnets.pytorchnet
from utils import set_seed, tb_writer, get_train_validation, dump_parameters
import settings


def build_model(model_name, num_classes, bias=True):
    model_name_lowered = model_name.lower()
    
    model = bagnet.bagnets.pytorchnet.bagnet9(pretrained=True).cuda()
    match model_name_lowered:
        case 'bagnet9':
            pass
        case 'bagnet17':
            model = bagnet.bagnets.pytorchnet.bagnet17(pretrained=True).cuda()
        case 'bagnet33':
            model = bagnet.bagnets.pytorchnet.bagnet33(pretrained=True).cuda()
        case _:
            raise Exception('Model not implemented!')
    model.fc = nn.Linear(model.fc.in_features, num_classes, bias=bias)

    return model


def train_or_test(model, dataloader, criterion, optimizer=None):
    is_train = optimizer is not None
    
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_loss = 0

    predictions = []

    loss = 0
    total_loss = 0

    for i, data in enumerate(tqdm(dataloader, unit='batch')):
        (image, label) = data

        input = image.to(settings.DEVICE)
        target = label.to(settings.DEVICE)

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        
        with grad_req:
            output = model(input)

            loss = criterion(output, target)

        predicted = torch.argmax(output.data, dim=1)
        if not is_train:
            predictions.extend(predicted.cpu().numpy())
        n_examples += target.size(0)
        n_correct += (predicted == target).sum().item()

        n_batches += 1
        total_loss += loss.item()

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted

    acc = n_correct / n_examples
    loss = total_loss / n_batches

    print(f'\tacc: \t\t{acc}')
    print(f'\tloss: \t\t{loss}')

    return acc, loss, predictions


def train():
    tb, dirname = tb_writer(settings.TENSORBOARD_LOGDIR_PREFIX, settings.MODEL_NAME)
    dump_parameters(tb.get_logdir(), 'settings.json')

    model = build_model(settings.MODEL_NAME, settings.NUM_CLASSES, settings.BASE_MODEL_BIAS)
    model = model.to(settings.DEVICE)
    model_multi = torch.nn.DataParallel(model)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model_multi.parameters(), lr=settings.LR, weight_decay=settings.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.NUM_EPOCHS)

    augment = v2.Compose([
        v2.Resize(256),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomRotation(degrees=(-15, 15)),
        v2.RandomPerspective(distortion_scale=0.3),
        v2.RandomCrop(settings.IMG_SIZE),
    ])

    trainset = datasets.ImageFolder(settings.TRAIN_DIR,
                                    transform=v2.Compose([
                                        augment,
                                        v2.ToImage(), 
                                        v2.ToDtype(torch.float32, scale=True),
                                        v2.Normalize(mean=settings.IMG_MEAN, std=settings.IMG_STD),
                                    ]))

    trainset, validationset = get_train_validation(trainset, use_percent=settings.DATA_PERCENTAGE, val_split=settings.VAL_SPLIT, stratify=True)

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=settings.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    validationloader = torch.utils.data.DataLoader(
        validationset, 
        batch_size=settings.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    best_acc = 0.0

    # Training:
    print('Training the model...')
    for epoch in range(settings.NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{settings.NUM_EPOCHS}')
        print(f'LR={scheduler.get_last_lr()}')

        # Train:
        print('Training:')
        model_multi.train()
        train_acc, train_loss, _ = train_or_test(model_multi, trainloader, criterion, optimizer)

        tb.add_scalar('Accuracy/train', train_acc, epoch)
        tb.add_scalar('Loss/train', train_loss, epoch)

        # Validation:
        print('Validation:')
        model_multi.eval()
        val_acc, val_loss, _ = train_or_test(model_multi, validationloader, criterion, None)

        tb.add_scalar('Accuracy/val', val_acc, epoch)
        tb.add_scalar('Loss/val', val_loss, epoch)
        tb.flush()

        if val_acc >= best_acc:
            best_acc = val_acc
            if val_acc >= settings.SAVE_MIN_ACC:
                torch.save(model_multi.module.state_dict(), os.path.join(settings.MODEL_PATH, 'bagnet', f'{dirname}.pth'))
                print(f"New best model (Accuracy) saved! Accuracy: {best_acc:.4f}")

        scheduler.step()
    
    tb.close()


def test(fname):
    model = build_model(settings.MODEL_NAME, settings.NUM_CLASSES, settings.BASE_MODEL_BIAS)
    model.load_state_dict(torch.load(fname, weights_only=True))
    model = model.to(settings.DEVICE)
    model_multi = torch.nn.DataParallel(model)

    testset = datasets.ImageFolder(settings.TEST_DIR,
                                   transform=v2.Compose([
                                       v2.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
                                       v2.ToImage(), 
                                       v2.ToDtype(torch.float32, scale=True),
                                       v2.Normalize(mean=settings.IMG_MEAN, std=settings.IMG_STD),
                                    ]))

    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=settings.BATCH_SIZE, 
        shuffle=False, 
        num_workers=8
    )

    criterion = torch.nn.CrossEntropyLoss()
    model_multi.eval()
    test_acc, test_loss, predictions = train_or_test(model_multi, testloader, criterion, None)
    return test_acc, test_loss, predictions


if __name__ == '__main__':
    settings.SEED = 123
    settings.DEVICE = 'cuda:0'
    
    torch.cuda.set_device(settings.DEVICE)
    print(torch.cuda.get_device_name(settings.DEVICE))
    set_seed(settings.SEED)

    # train()

    test('models/bagnet/bagnet17_2026.07.08_21-29-57.pth')
    