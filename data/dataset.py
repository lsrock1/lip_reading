from torch.utils.data import Dataset
import torch
from .preprocess import *
import os
from glob import glob
import cv2
import numpy as np


class LipreadingDataset(Dataset):
    """BBC Lip Reading dataset."""
    def __init__(self, directory, data_type, aug=True, landmark=False, landmark_seperate=False):
        self.file_list = sorted(glob(os.path.join(directory, '*', data_type, '*.mpg')))
        print('{} set: {}'.format(data_type, len(self.file_list)))
        self.label_list = getLabelFromFile(self.file_list)
        self.file_list = LandVideo(Video(self.file_list), data_type, landmark, landmark_seperate)
        self.labelToInt = labelToDict(self.label_list)
        self.landmark_seperate = landmark_seperate
        self.preprocess = Preprocess(aug)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        #load video into a tensor
        data = self.file_list[idx]
        label = self.label_list[idx]
        self.preprocess.reset()
        #temporalvolume = bbc(data[0] if self.landmarkloss or self.seperate else data, self.aug)
        #temporalvolume = data[0][:, :, 4:116, 4:116] if self.landmark_seperate else data[:, :, 4:116, 4:116]
        if self.landmark_seperate:
            return self.preprocess.process(data[0]), self.labelToInt[label], self.preprocess.process(data[1])
        else:
            return self.preprocess.process(data), self.labelToInt[label]


class LandVideo:
    def __init__(self, video, data_type, landmark = False, landmark_seperate=False):
        self.video = video
        self.landmark = landmark
        self.data_type = data_type
        self.landmark_seperate = landmark_seperate

    def __getitem__(self, key):
        data = self.video[key]
        if self.landmark:
            landmark_dir = self.video.getFile(key).split('.mpg')[0] + '/origin.npy'
            channel = np.zeros((1, data.shape[1], data.shape[2], data.shape[3]), dtype=np.uint8)
            for index, frame in enumerate(np.load(landmark_dir)):
                if frame[0, 0] != 0 and frame[0, 1] != 0:
                    for dot in frame:
                        channel[0, index, :, :] += make_gaussian((120, 120), center=(int(dot[0]/2) if int(dot[0]/2) < 120 else 119, int(dot[1]/2) if int(dot[1]/2) < 120 else 119))
                    #channel[0, index, int(dot[1]/2) if int(dot[1]/2) < 120 else 119, int(dot[0]/2) if int(dot[0]/2) < 120 else 119] = 255
            channel = channel
            if self.landmark_seperate:
                return data, channel
            else:
                data = np.concatenate([data, channel], axis=0)
        # if self.seperate:
        #     landmark_dir = self.video.getFile(key).split('.mpg')[0] + '/origin.npy'
        #     return data, torch.from_numpy(np.load(landmark_dir))
        return data

    def __len__(self):
        return len(self.video)


class Video(list):
    def __getitem__(self, key):
        return self.videoRead(super().__getitem__(key))

    def getFile(self, key):
        return super().__getitem__(key)

    def videoRead(self, item):
        tmp = []
        cap = cv2.VideoCapture(item)
        while(True):
            ret, frame = cap.read()
            if ret:
                # Our operations on the frame come here
                gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (120, 120)).reshape(-1, 1, 120, 120)
                tmp.append(gray)
                # Display the resulting frame
                #imshow(gray, cmap='gray')
            else:
                break
        cap.release()
        return np.concatenate(tmp, axis=1)

def getLabelFromFile(file_list):
    return [i.split('/')[-1].split('_')[0] for i in file_list]

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def labelToDict(label_list):
    return dict(zip(
        sorted(list(set(label_list))),
        list(range(len(label_list)))
    ))

def make_gaussian(size, sigma=30, center=None, d_type=np.uint8):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return (np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)*12.75).astype(d_type)