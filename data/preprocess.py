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
    
    def process(self, vidframes, type='image'):
        temporalvolume = torch.zeros(vidframes.shape[1], vidframes.shape[0], 112, 112)
        vidframes = np.transpose(vidframes, (1, 2, 3, 0))
        
        for index, data in enumerate(vidframes):
            result = transforms.Compose([
                transforms.ToPILImage(),
                self.croptransform,
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.40921722697548507 if type == 'image' else 0.15735664013013212,],
                    [0.15418997756505334 if type == 'image' else 0.2322109507292421,]),
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
