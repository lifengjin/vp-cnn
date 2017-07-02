import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch import autograd
import random

class CNN_Text(nn.Module):
    
    def __init__(self, args, char_or_word, vectors=None):
        super(CNN_Text,self).__init__()
        self.args = args

        if char_or_word == 'char':
            V = args.char_embed_num
            D = args.char_embed_dim
            Co = args.char_kernel_num
            Ks = args.char_kernel_sizes
        else:
            V = args.word_embed_num
            D = args.word_embed_dim
            Co = args.word_kernel_num
            Ks = args.word_kernel_sizes
        C = args.class_num
        Ci = 1
        self.embed = nn.Embedding(V, D, padding_idx=1)
        if vectors is not None:
            self.embed.weight.data = vectors

        # print(self.embed.weight.data[100])
        # print(self.embed.weight.data.size())
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        if char_or_word == 'word' or char_or_word == 'char':
            for layer in self.convs1:
                if args.ortho_init == True:
                    init.orthogonal(layer.weight.data)
                else:
                    layer.weight.data.uniform_(-0.01, 0.01)
                layer.bias.data.zero_()
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        if char_or_word == 'word' or char_or_word == 'char':
            if args.ortho_init == True:
                init.orthogonal(self.fc1.weight.data)
            else:
                init.normal(self.fc1.weight.data)
                self.fc1.weight.data.mul_(0.01)
            self.fc1.bias.data.zero_()
        # print(V, D, C, Ci, Co, Ks, self.convs1, self.fc1)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3) #(N,Co,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.confidence(x)
        logit = F.log_softmax(x) # (N,C)
        return logit

    def confidence(self, x):
        x = self.embed(x)  # (N,W,D)

        if self.args.static and self.args.word_vector:
            x = autograd.Variable(x.data)

        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        # print([x_p.size() for x_p in x])

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N,len(Ks)*Co)
        linear_out = self.fc1(x)
        return linear_out

class SimpleLogistic(nn.Module):
    def __init__(self, args):
        super(SimpleLogistic, self).__init__()
        self.args = args
        self.input_size = self.args.class_num * 2
        self.output_size = self.args.class_num
        self.layer_num = self.args.layer_num
        self.layers = nn.ModuleList([nn.Linear(self.input_size, self.input_size) if x < self.layer_num - 1 else
                       nn.Linear(self.input_size, self.output_size) for x in range(self.layer_num)])

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x

class StackingNet(nn.Module):
    def __init__(self, args):
        super(StackingNet, self).__init__()
        self.args = args
        self.params = nn.ParameterList([nn.Parameter(torch.rand(1)) for i in range(2)])

    def forward(self, inputs):
        output = 0
        for index, input in enumerate(inputs):
            output += input * self.params[index].expand(input.size())
        output = F.log_softmax(output)
        return output

class Memory:
    def __init__(self, mem_size, key_size):
        self.K = autograd.Variable(torch.zeros(key_size, mem_size))
        self.V = autograd.Variable(torch.zeros(mem_size).long())
        self.A = autograd.Variable(torch.zeros(mem_size).long())
        self.knn = 256
        self.query_indices = 0
        self.query_sims = 0
        self.x = 0

    def query(self, x):
        nn = torch.dot(x, self.K)
        self.query_sims, self.query_indices = torch.topk(nn, self.knn, dim=1)
        self.x = x
        return self.V[self.query_indices[:,0]]

    def loss_update(self, y, alpha=0.1):
        accuracy = self.V[self.query_indices[:,0]] == y
        # mask = self.query_indices[accuracy, 0]
        # self.K[mask] = torch.renorm(self.K[mask] + self.x.data)
        # self.A[mask] = 0
        for row_index, acc in enumerate(accuracy):
            self.A += 1
            if acc:
                positive_neighbor = self.query_sims[row_index, 0]
                negative_neighbor = self.query_sims[row_index][self.query_indices[row_index] != y[row_index]][0]
                loss = negative_neighbor - positive_neighbor + alpha if negative_neighbor - positive_neighbor + alpha > 0 else 0
                self.A[self.query_indices[row_index,0]] = 0
                self.K[self.query_indices[row_index,0]] = torch.renorm(self.K[self.query_indices[row_index,0]] + x.data[row_index])
            else:
                negative_neighbor = self.query_sims[row_index, 0]
                if not any(self.query_indices[row_index] == y[row_index]):
                    positive_neighbors = self.K[self.V == y[row_index]]
                    positive_neighbor = random.choice(positive_neighbors)
                    positive_neighbor = torch.dot(positive_neighbor, x[row_index])
                else:
                    positive_neighbor = self.query_sims[row_index][self.query_indices[row_index] == y[row_index]][0]
                _, oldest = torch.topk(self.A, 10)
                oldest_index = random.choice(oldest)
                self.K[oldest_index] = x.data[row_index]
                self.V[oldest_index] = y.data[row_index]
                self.A[oldest_index] = 0
                loss = negative_neighbor - positive_neighbor + alpha
        return loss




