from torch.utils.data import Dataset
from .preprocess import *
import os
from glob import glob
import cv2
import numpy as np


class LipreadingDataset(Dataset):
    """BBC Lip Reading dataset."""
    def __init__(self, directory, data_type, aug=True, landmark=False, landmarkloss=False, seperate=False):
        self.file_list = sorted(glob(os.path.join(directory, '*', data_type, '*.mpg')))
        print('{} set: {}'.format(data_type, len(self.file_list)))
        self.label_list = getLabelFromFile(self.file_list)
        self.file_list = LandVideo(Video(self.file_list), data_type, landmark, landmarkloss, seperate)
        self.labelToInt = labelToDict(self.label_list)
        self.aug = aug
        self.landmarkloss = landmarkloss
        self.seperate = seperate

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        #load video into a tensor
        data = self.file_list[idx]
        label = self.label_list[idx]
        temporalvolume = bbc(data[0] if self.landmarkloss or self.seperate else data, self.aug)
        if self.landmarkloss or self.seperate:
            return temporalvolume, self.labelToInt[label], data[1]
        return temporalvolume, self.labelToInt[label]


class LandVideo:
    def __init__(self, video, data_type, isLandmark = False, landmarkloss=False, seperate=False):
        self.video = video
        self.isLandmark = isLandmark
        self.landmarkloss = landmarkloss
        self.data_type = data_type
        self.seperate = seperate

    def __getitem__(self, key):
        data = self.video[key]
        if self.isLandmark:
            landmark_dir = self.video.getFile(key).split('.mpg')[0] + '/origin.npy'
            channel = np.zeros((1, data.shape[1], data.shape[2], data.shape[3]), dtype=np.int8)
            for index, frame in enumerate(np.load(landmark_dir)):
                for dot in frame:
                    channel[0, index, int(dot[1]/2) if int(dot[1]/2) < 120 else 119, int(dot[0]/2) if int(dot[0]/2) < 120 else 119] = 255
            data = np.concatenate([data, channel], axis=0)
        if (self.landmarkloss and self.data_type == 'train') or self.seperate:
            landmark_dir = self.video.getFile(key).split('.mpg')[0] + '/origin.npy'
            return data, torch.from_numpy(np.load(landmark_dir))
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