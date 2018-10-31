from __future__ import print_function
from models import LipRead
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
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
    parser.add_argument("-s", "--start", type=int,
                            help="start epoch if reload", default=1)
    parser.add_argument("-t", "--test", type=bool,
                            help="test mode", default=False)
    args = parser.parse_args()

    print("Loading options...")
    with open(args.config, 'r') as optionsFile:
        options = yaml.load(optionsFile.read())
    options['training']['start_epoch'] = args.start - 1
    if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
        print("Running cudnn benchmark...")
        torch.backends.cudnn.benchmark = True

    #Create the model.
    model = nn.DataParallel(LipRead(options).cuda())
    print('lr : ', options['training']['learning_rate'])
    print('weight_decay : ', options['training']['weight_decay'])
    optimizer = optim.Adam(
                model.parameters(),
                lr = options['training']['learning_rate'],
                # momentum = options['training']['momentum'],
                weight_decay = options['training']['weight_decay']
            )
    if options['training']['schedule']:
        scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones = options['training']['schedule'],
                    gamma = options['training']['lr_decay']
                )

    criterion = model.loss()

    if not os.path.isdir(os.path.join(options["general"]["save_path"], options['name'])):
        os.mkdir(os.path.join(options["general"]["save_path"], options['name']))
    if not os.path.isdir(os.path.join(options["general"]["save_path"], options['name'], 'optimizers')):
        os.mkdir(os.path.join(options["general"]["save_path"], options['name'], 'optimizers'))
    if not os.path.isdir(os.path.join(options["general"]["save_path"], options['name'], 'models')):
        os.mkdir(os.path.join(options["general"]["save_path"], options['name'], 'models'))
    
    if(options["general"]['model_load']):
        path = glob(os.path.join(options["general"]["save_path"], options['name'], 'models', 'model{}.pth'.format(args.start - 1)))
        #path = sorted(glob(os.path.join(options["general"]["save_path"], options['name'], 'models', '*.pth')), key=lambda name : int(name.split('/')[-1].replace('.pth', '').replace('model', '')))
        if path:
            print('load {} model..'.format(path[-1]))
            model.load_state_dict(torch.load(path[-1]))
        #path = sorted(glob(os.path.join(options["general"]["save_path"], options['name'], 'optimizers', '*.pth')), key=lambda name : int(name.split('/')[-1].replace('.pth', '').replace('optimizer', '')))
        path = glob(os.path.join(options["general"]["save_path"], options['name'], 'optimizers', 'optimizer{}.pth'.format(args.start - 1)))
        if path:
            print('load {} optimizer..'.format(path[-1]))
            optimizer.load_state_dict(torch.load(path[-1]))

    train_dataset = LipreadingDataset(
        options["training"]["data_path"], "train", 
        options['input']['aug'], options['model']['landmark'], 
        options['training']['landmarkloss'], options['model']['seperate'], options['model']['landmarkonly'])
    train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=options["input"]["batch_size"],
                        shuffle=options["input"]["shuffle"],
                        num_workers=options["input"]["num_worker"],
                        drop_last=True
                    )
    val_dataset = LipreadingDataset(options['validation']['data_path'],
                                    "val", False, options['model']['landmark'],
                                    False, options['model']['seperate'],
                                    options['model']['landmarkonly'])
    val_dataloader = DataLoader(
                                val_dataset,
                                batch_size=options["input"]["batch_size"],
                                shuffle=options["input"]["shuffle"],
                                num_workers=options["input"]["num_worker"],
                                drop_last=True
                            )
    batch_size = options["input"]["batch_size"]
    stats_frequency = options["training"]["stats_frequency"]
    if args.test:
        test_dataset = LipreadingDataset(options['validation']['data_path'],
                                        "test", False, options['model']['landmark'], 
                                        False, options['model']['seperate'],
                                        options['model']['landmarkonly'])
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=options["input"]["batch_size"],
            shuffle=options["input"]["shuffle"],
            num_workers=options["input"]["num_worker"],
            drop_last=True
        )
        model.eval()
        with torch.no_grad():
            print("Starting testing...")
            count = 0
            validator_function = model.validator_function()

            for i_batch, sample_batched in enumerate(test_dataloader):
                input = sample_batched[0]
                labels = sample_batched[1]

                input = input.cuda()
                labels = labels.cuda()
                
                outputs = model(input)

                count += validator_function(outputs, labels)

            accuracy = count / len(val_dataset)
            print('#############test result################')
            print('correct count: {}, total count: {}, accu: {}'.format(count, len(test_dataset), accuracy))
            # with open(os.path.join('./', options['name']+'.txt'), "a") as outputfile:
            #     outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}" .format(count, len(test_dataset), accuracy ))
            return
    


    for epoch in range(options["training"]["start_epoch"], options["training"]["max_epoch"]):

        model.train()
        if(options["training"]["train"]):
            if options['training']['schedule']:
                scheduler.step(epoch)
            running_loss = 0.0
            count = 0
            count_bs = 0
            startTime = datetime.now()
            print("Starting training...")
            for i_batch, sample_batched in enumerate(train_dataloader):
                optimizer.zero_grad()
                if options['training']['landmarkloss'] or options['model']['seperate']:
                    x = sample_batched[0].cuda()
                    labels = sample_batched[1].cuda()
                    dot_labels = sample_batched[2].float().cuda()
                else:
                    x = sample_batched[0].cuda()
                    labels = sample_batched[1].cuda()
                if not options['model']['seperate']:
                    outputs = model(x)
                else:
                    outputs = model(x, dot_labels)
                if options['training']['landmarkloss']:
                    outputs, dot = outputs
                    loss = criterion(outputs, labels, dot, dot_labels)
                else:
                    loss = criterion(outputs, labels)
                count += model.validator_function()(outputs, labels)
                count_bs += labels.shape[0]
                
                running_loss += loss.item()
                loss.backward()
                # for p,n in model.named_parameters():
                #     try:
                #         print('===========\ngradient:{}\n----------\n{}'.format(p,torch.sum(n.grad)))
                #     except:
                #         pass
                optimizer.step()
                if(i_batch % stats_frequency == 0 and i_batch != 0):
                    
                    print('[%d, %5d] loss: %.8f, acc: %f' %
                    (epoch + 1, i_batch + 1, running_loss / stats_frequency, count/count_bs))
                    try:
                        print('lr {}, name {}'.format(scheduler.get_lr(), options['name']))
                    except:
                        print('lr {}, name {}'.format(options['training']['learning_rate'], options['name']))
                    running_loss = 0.0
                    count = 0
                    count_bs = 0
                    currentTime = datetime.now()
                    output_iteration(i_batch * batch_size, currentTime - startTime, len(train_dataset))

        print("Epoch completed")
        if options['general']['model_save'] and options['training']['train']:
            print("saving state..")
            torch.save(model.state_dict(), os.path.join(options["general"]["save_path"], options['name'], 'models', 'model{}.pth'.format(epoch+1)))
            torch.save(optimizer.state_dict(), os.path.join(options["general"]["save_path"], options['name'], 'optimizers', 'optimizer{}.pth'.format(epoch+1)))

        model.eval()
        with torch.no_grad():
            if(options["validation"]["validate"]):
                print("Starting validation...")
                count = 0
                validator_function = model.validator_function()

                for i_batch, sample_batched in enumerate(val_dataloader):
                    if options['model']['seperate']:
                        x = sample_batched[0].cuda()
                        labels = sample_batched[1].cuda()
                        dot_labels = sample_batched[2].float().cuda()
                    else:
                        x = sample_batched[0].cuda()
                        labels = sample_batched[1].cuda()
                    if not options['model']['seperate']:
                        outputs = model(x)
                    else:
                        outputs = model(x, dot_labels)

                    count += validator_function(outputs, labels)

                accuracy = count / len(val_dataset)
                print('correct count: {}, total count: {}, accu: {}'.format(count, len(val_dataset), accuracy))
                with open(os.path.join('./', options['name']+'.txt'), "a") as outputfile:
                    outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}" .format(count, len(val_dataset), accuracy ))

if __name__ == '__main__':
    main()
