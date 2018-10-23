import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class NLLSequenceLoss(nn.Module):
    """
    Custom loss function.
    Returns a loss that is the sum of all losses at each time step.
    """
    def __init__(self):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        loss = 0.0
        # bs, length, 500
        # bs
        length = input.size(1)
        input = input.view(-1, 500)
        target = target.unsqueeze(1).expand(-1, length).view(-1)
        loss = self.criterion(input, target)
        # for i in range(0, 29):
        #     loss += self.criterion(input[i], target)

        return loss

def _validate(modelOutput, labels):
    averageEnergies = torch.sum(modelOutput.data, 1)
    maxvalues, maxindices = torch.max(averageEnergies, 1)
    count = 0

    for i in range(0, labels.size(0)):

        if maxindices[i] == labels[i]:
            count += 1

    return count


class LSTMBackend(nn.Module):
    def __init__(self, options):
        super(LSTMBackend, self).__init__()
        self.Module1 = nn.LSTM(input_size=options["model"]["input_dim"],
                                hidden_size=options["model"]["hidden_dim"],
                                num_layers=options["model"]["num_lstm"],
                                batch_first=True,
                                bidirectional=True)
        self.fc = nn.Linear(options["model"]["hidden_dim"] * 2,
                                options["model"]["num_class"])
        self.loss = NLLSequenceLoss()
        self.validator = _validate

    def forward(self, input):
        input, _ = self.Module1(input)
        input = self.fc(input)

        return input
