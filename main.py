from __future__ import print_function
from models import LipRead
import torch
from training import Trainer
from validation import Validator
from optimizer import get_optimizer
from glob import glob
import yaml
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str,
                        help="configuration file")
args = parser.parse_args()

print("Loading options...")
with open(args.config, 'r') as optionsFile:
    options = yaml.loads(optionsFile.read())

if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

#Create the model.
model = LipRead(options).cuda()
optimizer = get_optimizer(model, options)

if(options["general"]['model_load']):
    path = sorted(glob(os.path.join(options["general"]["save_path"], 'models', '*.pth')), key=lambda name : int(name.replace('.pth').replace('model')))[-1]
    if path:
        model.load_state_dict(torch.load(path))
    path = sorted(glob(os.path.join(options["general"]["save_path"], 'optimizers', '*.pth')), key=lambda name : int(name.replace('.pth').replace('model')))[-1]
    if path:
        optimizer.load_state_dict(torch.load(path))

trainer = Trainer(model, optimizer, options)
validator = Validator(model, options)

for epoch in range(options["training"]["start_epoch"], options["training"]["max_epoch"]):

    if(options["training"]["train"]):
        trainer.epoch(epoch)

    if(options["validation"]["validate"]):
        validator.epoch(model)
