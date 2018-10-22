import torch.optim as optim

def get_optimizer(model, options):
    return optim.SGD(
            model.parameters(),
            lr = options['training']['learning_rate'],
            momentum = options['training']['momentum'],
            weight_decay = options['training']['weight_decay']
        )