from torch.utils.data import Dataset
import cv2
import numpy as np


class HandGestureDataset(Dataset):
    def __init__(self, data_dir, df, type_of_image=None, classes=None, transform=None):
        self.classes = classes
        self.data_dir = data_dir
        self.transform = transform

        self.binary_fps = df['binary'].tolist()
        self.gray_fps = df['grayscale'].tolist()
        self.label_list = df['label'].tolist()
        self.type_of_image = type_of_image

    def __len__(self):
        return len(self.binary_fps)

    def __getitem__(self, idx):
        binary = cv2.imread(self.binary_fps[idx], 0)  # automatically reads in as RGB (required by vgg16)
        gray = cv2.imread(self.gray_fps[idx], 0)
        label = self.label_list[idx]

        if self.type_of_image == 'binary':
            image = binary
        elif self.type_of_image == 'grayscale':
            image = gray
        elif self.type_of_image == 'bin_gray':  # binary mask applied on grayscale image to extract the hand
            image = np.multiply(gray, binary)
        else:
            raise ValueError('{} is not one of binary, grayscale, or bin_gray'.format(self.type_of_image))

        # two_channel_img = np.zeros((binary.shape[0], binary.shape[1], 2))
        # two_channel_img[:, :, 0] = binary
        # two_channel_img[:, :, 1] = gray
        # two_channel_img = two_channel_img.astype(int)

        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}

        return sample