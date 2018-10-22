from torch.utils.data import Dataset
from .preprocess import *
import os
from glob import glob


class LipreadingDataset(Dataset):
    """BBC Lip Reading dataset."""
    def __init__(self, directory, set, augment=True):
        self.label_list, self.file_list = self.build_file_list(directory, set)
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        #load video into a tensor
        label, filename = self.file_list[idx]
        vidframes = load_video(filename)
        temporalvolume = bbc(vidframes, self.augment)

        sample = {'temporalvolume': temporalvolume, 'label': torch.LongTensor([label])}

        return sample


class LandVideo:
    def __init__(self, video, isLandmark = False):
        self.video = video
        self.isLandmark = isLandmark

    def __getitem__(self, key):
        data = self.video[key]
        if self.isLandmark:
            splitName = self.video.getFile(key).split('/')
            filename = splitName[-1][:-4]
            landmarks = sorted(glob(os.path.join('/'.join(splitName[:-2]), 'landmark', 'origin', '*', filename, '*')), key=natural_keys)
            channel = np.zeros([1, data.shape[1], data.shape[2], data.shape[3]])
            for i in range(data.shape[1]):
                tmp = np.load(landmarks[i])
                for dot in tmp:
                    try:
                        channel[0, i, dot[1], dot[0]] = 1
                    except:
                        pass
            data = np.concatenate([data, channel], axis=0)
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


class Script(list):
    def __getitem__(self, key):
        return self.scriptRead(super().__getitem__(key))

    def scriptRead(self, item):
        tmp = ''
        result = []
        with open(item, 'r') as f:
            for line in f.readlines():
                line = line.strip().upper()
                if line != '':
                    tmp += line.split()[-1]
                tmp += ' '
        tmp = tmp.rstrip().replace('SIL', '').strip()
        for i in tmp:
            result.append(wtoi[i])

        return np.array(result, dtype='int')


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]