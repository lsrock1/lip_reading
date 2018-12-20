import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip
import numpy as np

class Preprocess:

    def __init__(self, aug):
        self.aug = aug
        if aug:
            crop = StatefulRandomCrop((120, 120), (112, 112))
            flip = StatefulRandomHorizontalFlip(0.5)

            self.croptransform = transforms.Compose([
                crop,
                flip
            ])
        else:
            self.croptransform = transforms.CenterCrop((112, 112))

    def reset(self):
        if self.aug:
            crop = StatefulRandomCrop((120, 120), (112, 112))
            flip = StatefulRandomHorizontalFlip(0.5)
            self.croptransform = transforms.Compose([
                    crop,
                    flip
                ])
        else:
            self.croptransform = transforms.CenterCrop((112, 112))
    
    def process(self, vidframes):
        temporalvolume = torch.zeros(vidframes.shape[1], vidframes.shape[0], 112, 112)
        vidframes = np.transpose(vidframes, (1, 2, 3, 0))
        
        for index, data in enumerate(vidframes):
        result = transforms.Compose([
            transforms.ToPILImage(),
            self.croptransform,
            transforms.ToTensor(),
            transforms.Normalize(
                [0.40921722697548507,],[0.15418997756505334,]),
        ])(np.expand_dims(data[:, :, 0], axis=2))

        temporalvolume[index, 0, :, :] = result
        if data.shape[2] == 2:
            result = transforms.Compose([
                transforms.ToPILImage(),
                self.croptransform,
                transforms.ToTensor(),
                transforms.Normalize([0.15735664013013212,],[0.2322109507292421,]),
            ])(np.expand_dims(data[:, :, 1], axis=2))

            temporalvolume[index, 1, :, :] = result
    return temporalvolume.transpose(0, 1)

def bbc(vidframes, augmentation=True, seperate=False):
    """Preprocesses the specified list of frames by center cropping.
    This will only work correctly on videos that are already centered on the
    mouth region, such as LRITW.

    Args:
        vidframes :  The frames of the video as a list of
            4D numpy (channels, Frame, height, width)

    Returns:
        FloatTensor: The video as a temporal volume, represented as a 5D tensor
            (batch, channel, time, width, height)"""
    temporalvolume = torch.zeros(vidframes.shape[1], vidframes.shape[0], 112, 112)
    vidframes = np.transpose(vidframes, (1, 2, 3, 0))
    # frame, height, width, channel
    croptransform = transforms.CenterCrop((112, 112))
    
    if(augmentation):
        crop = StatefulRandomCrop((120, 120), (112, 112))
        flip = StatefulRandomHorizontalFlip(0.5)

        croptransform = transforms.Compose([
            crop,
            flip
        ])

    for index, data in enumerate(vidframes):
        result = transforms.Compose([
            transforms.ToPILImage(),
            croptransform,
            transforms.ToTensor(),
            transforms.Normalize(
                [0.40921722697548507,],[0.15418997756505334,]),
        ])(np.expand_dims(data[:, :, 0], axis=2))

        temporalvolume[index, 0, :, :] = result
        if data.shape[2] == 2:
            result = transforms.Compose([
                transforms.ToPILImage(),
                croptransform,
                transforms.ToTensor(),
                transforms.Normalize([0.15735664013013212,],[0.2322109507292421,]),
            ])(np.expand_dims(data[:, :, 1], axis=2))

            temporalvolume[index, 1, :, :] = result
    return temporalvolume.transpose(0, 1)
