from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
import os
import math

def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{} hrs, {} mins, {} secs".format(hours, minutes, seconds)

def output_iteration(i, time, totalitems):
    os.system('clear')

    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)

    print("Iteration: {}\nElapsed Time: {} \nEstimated Time Remaining: {}".format(i, timedelta_string(time), timedelta_string(estTime)))

class Trainer():
    def __init__(self, model, optimizer, options):
        self.dataset = LipreadingDataset(options["training"]["data_path"], "train")
        self.dataloader = DataLoader(
                            self.dataset,
                            batch_size=options["input"]["batchsize"],
                            shuffle=options["input"]["shuffle"],
                            num_workers=options["input"]["numworkers"],
                            drop_last=True
                        )
        self.usecudnn = options["general"]["usecudnn"]
        self.batch_size = options["input"]["batchsize"]
        self.stats_frequency = options["training"]["statsfrequency"]
        self.save_path = options['general']['save_path']

        self.optimizer = optimizer
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones = options['training']['schedule'],
            gamma = options['training']['lr_decay']
        )
        self.criterion = model.loss().cuda()

    def epoch(self, model, epoch):
        #set up the loss function.
        self.scheduler.step()

        startTime = datetime.now()
        print("Starting training...")
        for i_batch, sample_batched in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            input = sample_batched['temporalvolume']
            labels = sample_batched['label']

            if(self.usecudnn):
                input = input.cuda()
                labels = labels.cuda()

            outputs = model(input)
            loss = self.criterion(outputs, labels.squeeze(1))

            loss.backward()
            optimizer.step()
            sampleNumber = i_batch * self.batch_size

            if(sampleNumber % self.stats_frequency == 0):
                currentTime = datetime.now()
                output_iteration(sampleNumber, currentTime - startTime, len(self.dataset))

        print("Epoch completed, saving state...")
        torch.save(model.state_dict(), os.path.join(self.save_path, 'model{}.pth'.format(epoch)))
        torch.save(self.optimizer.state_dict(), os.path.join(self.save_path, 'optimizer{}.pth'.format(epoch)))
