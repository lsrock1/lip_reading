from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
import os

class Validator():
    def __init__(self, options):
        self.dataset = LipreadingDataset(options['validation']['data_path'],
                                    "val", False)
        self.dataloader = DataLoader(
                                    self.validationdataset,
                                    batch_size=options["input"]["batch_size"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["num_worker"],
                                    drop_last=True
                                )
        self.batch_size = options["input"]["batch_size"]

    def epoch(self, model):
        print("Starting validation...")
        count = 0
        validator_function = model.validator_function()

        for i_batch, sample_batched in enumerate(self.dataloader):
            input = sample_batched['temporalvolume']
            labels = sample_batched['label']

            if(self.usecudnn):
                input = input.cuda()
                labels = labels.cuda()

            outputs = model(input)

            count += validator_function(outputs, labels)

            print(count)

        accuracy = count / len(self.dataset)
        with open(ps.path.join(options['validation']['data_path'], options['name']+'.txt'), "a") as outputfile:
            outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}" .format(count, len(self.dataset), accuracy ))
