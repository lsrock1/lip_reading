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
        self.dot_crit = nn.MSELoss()

    def forward(self, input, target, dot=False, dot_labels=False):
        loss = 0.0
        # bs, length, 500
        # bs
        length = input.size(1)
        input = input.view(-1, 500)
        target = target.unsqueeze(1).expand(-1, length).contiguous().view(-1)
        loss = self.criterion(input, target)
        if not isinstance(dot_labels, bool):
            loss += self.dot_crit(dot*125 + 125, dot_labels)
        # for i in range(0, 29):
        #     loss += self.criterion(input[i], target)

        return loss

def _validate(x, labels):
    x = torch.sum(F.log_softmax(x, dim=2), dim=1)
    _, maxindices = torch.max(x, dim=1)
    count = 0

    for i in range(0, labels.size(0)):
        if maxindices[i] == labels[i]:
            count += 1

    return count


class LSTMBackend(nn.Module):
    def __init__(self, options):
        super(LSTMBackend, self).__init__()
        self.lstm = nn.LSTM(input_size=options["model"]["input_dim"],
                                hidden_size=options["model"]["hidden_dim"],
                                num_layers=options["model"]["num_lstm"],
                                batch_first=True,
                                bidirectional=True)
        self.fc = nn.Linear(
            (options["model"]["hidden_dim"], options["model"]["num_class"]))
        self.loss = NLLSequenceLoss()
        self.validator = _validate

    def forward(self, x):
        return self.fc(self.lstm(x)[0])
