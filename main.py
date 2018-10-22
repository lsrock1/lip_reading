from __future__ import print_function
from models import LipRead
import torch
from training import Trainer
from validation import Validator
from optimizer import get_optimizer
from glob import glob
import yaml
import argparse
import os

def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{} hrs, {} mins, {} secs".format(hours, minutes, seconds)

def output_iteration(i, time, totalitems):

    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)

    print("Iteration: {}\nElapsed Time: {} \nEstimated Time Remaining: {}".format(i, timedelta_string(time), timedelta_string(estTime)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                            help="configuration file")
    args = parser.parse_args()

    print("Loading options...")
    with open(args.config, 'r') as optionsFile:
        options = yaml.load(optionsFile.read())

    if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
        print("Running cudnn benchmark...")
        torch.backends.cudnn.benchmark = True

    #Create the model.
    model = LipRead(options).cuda()
    optimizer = optim.SGD(
                model.parameters(),
                lr = options['training']['learning_rate'],
                momentum = options['training']['momentum'],
                weight_decay = options['training']['weight_decay']
            )

    scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones = options['training']['schedule'],
                gamma = options['training']['lr_decay']
            )

    criterion = model.loss()

    if(options["general"]['model_load']):
        path = sorted(glob(os.path.join(options["general"]["save_path"], 'models', '*.pth')), key=lambda name : int(name.replace('.pth').replace('model')))
        if path:
            model.load_state_dict(torch.load(path[-1]))
        path = sorted(glob(os.path.join(options["general"]["save_path"], 'optimizers', '*.pth')), key=lambda name : int(name.replace('.pth').replace('model')))
        if path:
            optimizer.load_state_dict(torch.load(path[-1]))

    train_dataset = LipreadingDataset(options["training"]["data_path"], "train", True, options['model']['landmark'])
    train_dataloader = DataLoader(
                        dataset,
                        batch_size=options["input"]["batch_size"],
                        shuffle=options["input"]["shuffle"],
                        num_workers=options["input"]["num_worker"],
                        drop_last=True
                    )
    val_dataset = LipreadingDataset(options['validation']['data_path'],
                                    "val", False, options['model']['landmark'])
    val_dataloader = DataLoader(
                                self.dataset,
                                batch_size=options["input"]["batch_size"],
                                shuffle=options["input"]["shuffle"],
                                num_workers=options["input"]["num_worker"],
                                drop_last=True
                            )
    batch_size = options["input"]["batch_size"]
    stats_frequency = options["training"]["stats_frequency"]

    for epoch in range(options["training"]["start_epoch"], options["training"]["max_epoch"]):

        model.train()
        if(options["training"]["train"]):
            scheduler.step()
            running_loss = 0.0
            startTime = datetime.now()
            print("Starting training...")
            for i_batch, sample_batched in enumerate(train_dataloader):
                optimizer.zero_grad()
                input = sample_batched[0].cuda()
                labels = sample_batched[1].cuda()
                outputs = model(input)

                loss = criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                if(i_batch % stats_frequency == 0):
                    print('[%d, %5d] loss: %.8f' %
                    (epoch + 1, i_batch + 1, running_loss / stats_frequency))
                    running_loss = 0.0
                    currentTime = datetime.now()
                    output_iteration(i_batch * batch_size, currentTime - startTime, len(dataset))

        print("Epoch completed, saving state...")
        torch.save(model.state_dict(), os.path.join(options["general"]["save_path"], 'model{}.pth'.format(epoch)))
        torch.save(optimizer.state_dict(), os.path.join(options["general"]["save_path"], 'optimizer{}.pth'.format(epoch)))

        model.eval()
        with torch.no_grad():
            if(options["validation"]["validate"]):
                print("Starting validation...")
                count = 0
                validator_function = model.validator_function()

                for i_batch, sample_batched in enumerate(val_dataloader):
                    input = sample_batched[0]
                    labels = sample_batched[1]

                    input = input.cuda()
                    labels = labels.cuda()
                    
                    outputs = model(input)

                    count += validator_function(outputs, labels)

                    print(count)

                accuracy = count / len(val_dataset)
                with open(ps.path.join(options['validation']['data_path'], options['name']+'.txt'), "a") as outputfile:
                    outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}" .format(count, len(val_dataset), accuracy ))

if __name__ == '__main__':
    main()
