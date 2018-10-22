import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip
import numpy as np

def bbc(vidframes, augmentation=True):
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
            transforms.CenterCrop((122, 122)),
            croptransform,
            transforms.ToTensor(),
        ])(data)
        temporalvolume[index] = result
        print(data.shape)
        print(temporalvolume[index].size())
        break
    return temporalvolume.transpose(0, 1)
