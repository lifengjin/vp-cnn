from cnn_classifier.model import CNN_Text, Memory
from torch import nn
import torch.nn.functional as F
import torch
from utils.future_ops import normalize

class bi_CNN_Text(nn.Module):
    def __init__(self, *args, **kwargs):
        super(bi_CNN_Text, self).__init__()
        self.args = args
        self.cnn = CNN_Text(*args, **kwargs)

    def forward(self, x):
        s1, s2 = x
        s1 = self.cnn.confidence(s1)
        s2 = self.cnn.confidence(s2)
        return s1, s2

    def compute_similarity(self, x):
        s1, s2 = self.forward(x)
        # y = F.pairwise_distance(s1, s2, p=1).view(-
        # print(s1.size(), s2.size())
        s = s1 * s2
        y = torch.sum(s, 1)
        # print(y.size())
        return y

class CNN_Mem(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CNN_Mem, self).__init__()
        self.args = args
        self.memory = Memory(kwargs.pop('mem_size'), kwargs.pop('key_size'))
        self.cnn = CNN_Text(*args, **kwargs)

    def forward(self, x, y, update=True):
        x = self.cnn.confidence(x)
        x = normalize(x)
        loss, accuracy = self.memory.compute_loss(x, y, update=update)
        return loss, accuracy

    def update_mem(self):
        self.memory.update()

    def cuda(self):
        self.cnn.cuda()
        self.memory.cuda()