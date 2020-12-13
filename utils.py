# define a list of functions that will be used in main

import os
import cv2

import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from HandGestureDataset import HandGestureDataset


def get_padding(img, max_h=588, max_w=352):
    c, h, w = img.shape

    # drop the very last column/row to make this image even
    if h % 2 == 1:
        img = img[:-1, :]
    if w % 2 == 1:
        img = img[:, :-1]

    h_padding = (max_h - h) / 2
    w_padding = (max_w - w) / 2
    padding = (int(w_padding), int(w_padding), int(h_padding), int(h_padding))
    return padding


class NewPad(object): # to be called by transforms.compose
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill  # the value to fill with
        self.padding_mode = padding_mode

    def __call__(self, img):  # this is required to make this object suitable for the compose pipeline
        return F.pad(img, get_padding(img), mode=self.padding_mode, value=self.fill)


def prepare_csv(data_dir, classes=[1, 2, 3]):
    # read all image filepaths (not the images themselves)
    img_list = []
    lab_list = []

    for t in ['binary', 'grayscale']:
        img_list_for_t = []
        for c in classes:
            img_path = os.path.join(data_dir, 'gesture_{}'.format(c), t)
            img_dir = os.listdir(img_path)
            img_dir.sort()
            num_imgs_in_class = len(img_dir)
            for i in range(num_imgs_in_class):
                img_file = img_dir[i]
                if not img_file.startswith('.'):
                    img_list_for_t.append(os.path.join(img_path, img_file))
                    lab_list.append(c - 1)  # instead of 1, 2, and 3, let's subtract 1 to make it 0-indexed
        img_list.append(img_list_for_t)

    img_list = np.array(img_list)
    lab_list = np.array(lab_list)

    shuffler = np.random.permutation(img_list.shape[1])
    img_list = img_list[:, shuffler]
    lab_list = lab_list[shuffler]

    df = pd.DataFrame({'binary': img_list[0, :], 'grayscale': img_list[1, :], 'label': lab_list})

    # now we split
    total_num = img_list.shape[1]
    df_train, df_valid, df_test = df[:int(total_num * 0.7)], df[int(total_num * 0.7):int(total_num * 0.85)], df[int(
        total_num * 0.85):]

    return df_train, df_valid, df_test


def prepare_datasets(my_seed=30, type_of_image='binary'):
    data_dir = '/content/drive/MyDrive/Umich Classes/EECS504/EECS504 Project/Data/'
    OUR_IMAGE_HEIGHT = 320
    OUR_IMAGE_WIDTH = 192 # 1.67
    FINAL_CONV_OUTPUT_SIZE = (int(OUR_IMAGE_HEIGHT/32), int(OUR_IMAGE_WIDTH/32))

    assert OUR_IMAGE_HEIGHT % 32 is 0, "OUR_IMAGE_HEIGHT must be divisible by 32"
    assert OUR_IMAGE_WIDTH % 32 is 0, "OUR_IMAGE_WIDTH must be divisible by 32"

    np.random.seed(seed=my_seed)  # set this seed in main

    classes = [1, 2, 3] # digits to classify

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine((-20, 20), translate=(0.1, 0.1), scale=(0.5, 1.3)), # requires images to be converted to PIL first
        transforms.ToTensor(),
        NewPad(), # requires images to be tensors; pad from the center out to make 608 x 352
        transforms.Resize([OUR_IMAGE_HEIGHT, OUR_IMAGE_WIDTH]), # make it a little smaller so that the network requires less memory
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5), (0.5)),
        ]
    )

    # for validation -- not gonna apply the data augmentation steps
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        NewPad(), # requires images to be tensors; pad from the center out to make 608 x 352
        transforms.Resize([OUR_IMAGE_HEIGHT, OUR_IMAGE_WIDTH]), # make it a little smaller so that the network requires less memory
        transforms.Normalize((0.5), (0.5)),
        ]
    )

    df_train, df_valid, df_test = prepare_csv(data_dir, classes = classes)

    train_dataset = HandGestureDataset(data_dir, df_train, type_of_image=type_of_image, classes=classes, transform=train_transform)
    valid_dataset = HandGestureDataset(data_dir, df_valid, type_of_image=type_of_image, classes=classes, transform=valid_transform)
    test_dataset = HandGestureDataset(data_dir, df_test, type_of_image=type_of_image, classes=classes, transform=valid_transform)

    return train_dataset, valid_dataset, test_dataset

