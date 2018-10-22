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

    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)

    print("Iteration: {}\nElapsed Time: {} \nEstimated Time Remaining: {}".format(i, timedelta_string(time), timedelta_string(estTime)))

class Trainer():
    def __init__(self, model, optimizer, options):
        self.dataset = LipreadingDataset(options["training"]["data_path"], "train", True, options['model']['landmark'])
        self.dataloader = DataLoader(
                            self.dataset,
                            batch_size=options["input"]["batch_size"],
                            shuffle=options["input"]["shuffle"],
                            num_workers=options["input"]["num_worker"],
                            drop_last=True
                        )
        self.usecudnn = options["general"]["usecudnn"]
        self.batch_size = options["input"]["batch_size"]
        self.stats_frequency = options["training"]["stats_frequency"]

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
        running_loss = 0.0
        startTime = datetime.now()
        print("Starting training...")
        for i_batch, sample_batched in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            input = sample_batched[0]
            labels = sample_batched[1]
            if(self.usecudnn):
                input = input.cuda()
                labels = labels.cuda()
            outputs = model(input)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            if(i_batch % self.stats_frequency == 0):
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i_batch + 1, running_loss / self.stats_frequency))
                running_loss = 0.0
                currentTime = datetime.now()
                output_iteration(i_batch * self.batch_size, currentTime - startTime, len(self.dataset))
