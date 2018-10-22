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
optimizer = get_optimizer(model, options)

if(options["general"]['model_load']):
    path = sorted(glob(os.path.join(options["general"]["save_path"], 'models', '*.pth')), key=lambda name : int(name.replace('.pth').replace('model')))
    if path:
        model.load_state_dict(torch.load(path[-1]))
    path = sorted(glob(os.path.join(options["general"]["save_path"], 'optimizers', '*.pth')), key=lambda name : int(name.replace('.pth').replace('model')))
    if path:
        optimizer.load_state_dict(torch.load(path[-1]))

trainer = Trainer(model, optimizer, options)
validator = Validator(options)

for epoch in range(options["training"]["start_epoch"], options["training"]["max_epoch"]):

    model.train()
    if(options["training"]["train"]):
        trainer.epoch(model, epoch)

    print("Epoch completed, saving state...")
    torch.save(model.state_dict(), os.path.join(options["general"]["save_path"], 'model{}.pth'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(options["general"]["save_path"], 'optimizer{}.pth'.format(epoch)))

    model.eval()
    with torch.no_grad():
        if(options["validation"]["validate"]):
            validator.epoch(model)
