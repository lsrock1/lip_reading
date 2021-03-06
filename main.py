from __future__ import print_function
from models import LipRead
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
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
    parser.add_argument("-b", "--best", type=float,
    help="best epoch val accuracy", default=0.)
    parser.add_argument("-e", "--bepoch", type=int,
    help="the number of bad epoch", default=0)
    args = parser.parse_args()
    writer = SummaryWriter()
    
    print("Loading options...")
    with open(args.config, 'r') as optionsFile:
        options = yaml.load(optionsFile.read())
    options['training']['start_epoch'] = args.start - 1
    if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
        print("Running cudnn benchmark...")
        torch.backends.cudnn.benchmark = True
    writer.add_text('name', options['name'])
    #Create the model.
    model = LipRead(options).cuda()
    print('lr : ', options['training']['learning_rate'])
    print('weight_decay : ', options['training']['weight_decay'])
    print('landmark : ', options['input']['landmark'])
    print('landmark channel concat : ', not options['input']['landmark_seperate'])
    print('coord conv : ', options['model']['coord'])
    print('attention : ', options['model']['attention'])
    optimizer = optim.Adam(
                model.parameters(),
                lr = options['training']['learning_rate'],
                weight_decay = options['training']['weight_decay']
            )
    if options['training']['schedule'] == 'plateau':
        if args.start > 1 and args.best == 0. and not args.test:
            print("must have best accuracy")
            raise
        plat = True
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=options['training']['lr_decay'],
            patience=3,
            verbose=True,
            threshold=0.003,
            threshold_mode='abs',
        )
        scheduler.best = args.best
        scheduler.num_bad_epochs = args.bepoch
    else:
        plat = False
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
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = options['training']['learning_rate']

    train_dataset = LipreadingDataset(
        options["training"]["data_path"], "train", 
        options['input']['aug'], 
        options['input']['landmark'], 
        options['input']['landmark_seperate'])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=options["input"]["batch_size"],
        shuffle=options["input"]["shuffle"],
        num_workers=options["input"]["num_worker"],
        drop_last=True
        )
    val_dataset = LipreadingDataset(
        options['validation']['data_path'], "val", 
        False, 
        options['input']['landmark'], 
        options['input']['landmark_seperate'])
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=options["input"]["batch_size"],
        shuffle=options["input"]["shuffle"],
        num_workers=options["input"]["num_worker"],
        drop_last=True
        )
    train_size = len(train_dataset)
    
    batch_size = options["input"]["batch_size"]
    stats_frequency = options["training"]["stats_frequency"]
    if args.test:
        test_dataset = LipreadingDataset(
            options['validation']['data_path'], "test", 
            False, 
            options['input']['landmark'], 
            options['input']['landmark_seperate']
            )
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
            if options['training']['schedule'] and not plat:
                scheduler.step(epoch)
            running_loss = 0.0
            count = 0
            count_bs = 0
            startTime = datetime.now()
            print("Starting training...")
            for i_batch, sample_batched in enumerate(train_dataloader):
                optimizer.zero_grad()
                if options['input']['landmark_seperate']:
                    x = sample_batched[0].cuda()
                    labels = sample_batched[1].cuda()
                    dot_labels = sample_batched[2].float().cuda()
                else:
                    x = sample_batched[0].cuda()
                    labels = sample_batched[1].cuda()
                if not options['input']['landmark_seperate']:
                    outputs = model(x)
                else:
                    outputs = model(x, dot_labels)
                loss = criterion(outputs, labels)
                count += model.validator_function()(outputs, labels)
                count_bs += labels.shape[0]
                
                running_loss += loss.item()
                loss.backward()

                optimizer.step()
                if(i_batch % stats_frequency == 0 and i_batch != 0):
                    
                    print('[%d, %5d] loss: %.8f, acc: %f' %
                    (epoch + 1, i_batch + 1, running_loss / stats_frequency, count/count_bs))
                    writer.add_scalar('train/loss', running_loss/stats_frequency, (epoch-1) * train_size + i_batch*batch_size)
                    writer.add_scalar('train/accuracy', count/count_bs, (epoch-1) * train_size + i_batch*batch_size)
                    writer.add_scalar('train/true', count, (epoch-1) * train_size + i_batch*batch_size)
                    writer.add_scalar('train/false', count_bs-count, (epoch-1) * train_size + i_batch*batch_size)
                    try:
                        print('lr {}, name {}'.format(optimizer.param_groups[-1]['lr'], options['name']))
                        writer.add_scalar('train/lr', optimizer.param_groups[-1]['lr'], (epoch-1) * train_size + i_batch*batch_size)
                    except Exception as e:
                        print('lr {}, name {}'.format(options['training']['learning_rate'], options['name']))
                        writer.add_scalar('train/lr', options['training']['learning_rate'], (epoch-1) * train_size + i_batch*batch_size)
                    running_loss = 0.0
                    count = 0
                    count_bs = 0
                    currentTime = datetime.now()
                    output_iteration(i_batch * batch_size, currentTime - startTime, train_size)

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
                    if options['input']['landmark_seperate']:
                        x = sample_batched[0].cuda()
                        labels = sample_batched[1].cuda()
                        dot_labels = sample_batched[2].float().cuda()
                    else:
                        x = sample_batched[0].cuda()
                        labels = sample_batched[1].cuda()
                    if not options['input']['landmark_seperate']:
                        outputs = model(x)
                    else:
                        outputs = model(x, dot_labels)

                    count += validator_function(outputs, labels)

                accuracy = count / len(val_dataset)
                if options['training']['schedule'] and plat:
                    scheduler.step(accuracy)
                writer.add_scalar('val/accuracy', accuracy, epoch)
                writer.add_scalar('val/true', count, epoch)
                writer.add_scalar('val/false', len(val_dataset)-count, epoch)
                print('correct count: {}, total count: {}, accu: {}'.format(count, len(val_dataset), accuracy))
                with open(os.path.join('./', options['name']+'.txt'), "a") as outputfile:
                    outputfile.write("\ncorrect count: {}, total count: {} accuracy: {}" .format(count, len(val_dataset), accuracy ))

if __name__ == '__main__':
    main()
