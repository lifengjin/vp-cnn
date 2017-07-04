import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch import autograd
import random
from utils.future_ops import normalize

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
    def __init__(self, mem_size=1000, key_size=300):
        self.K = autograd.Variable(normalize(torch.ones(mem_size, key_size)).t())
        # self.K = torch.norm(self.K, 2, 0).expand_as(self.K)
        self.V = autograd.Variable(torch.zeros(mem_size).long())
        self.A = autograd.Variable(torch.zeros(mem_size).long())
        self.knn = 256
        self.query_indices = 0
        self.query_sims = 0
        self.x = 0
        self.index_vector_target_update = []

    def query(self, x):
        self.index_vector_target_update = []
        nn = torch.mm(x, self.K)
        # print('similarity scores are {} '.format(nn))
        self.query_sims, self.query_indices = torch.topk(nn, self.knn, dim=1)
        # print('top K similarity scores are {} and the indices are {} '.format(self.query_sims, self.query_indices))
        self.x = x
        # print('the predicted label is {}'.format(self.query_indices[:,0]))
        return self.V[self.query_indices[:,0].data]

    def compute_loss(self, x, y, alpha=0.1, update=True):
        self.index_vector_target_update = []
        # print('x is {}'.format(x))
        # print('K is {}'.format(self.K))
        self.query(x)
        accuracy = self.V[self.query_indices[:,0].data] == y
        loss = autograd.Variable(torch.zeros(1).cuda())
        if not update:
            return loss, accuracy
        # print('accuracy',accuracy)
        # mask = self.query_indices[accuracy, 0]
        # self.K[mask] = torch.renorm(self.K[mask] + self.x.data)
        # self.A[mask] = 0
        for row_index, acc in enumerate(accuracy):
            self.A += 1
            # print(acc, row_index)
            if acc.data[0] == 1:
                # print('predicted label is right.')
                positive_neighbor = self.query_sims[row_index, 0]
                # print(self.query_sims[row_index], self.V[self.query_indices[row_index].data].data != y[row_index].data[0])
                negative_neighbor = self.query_sims[row_index][self.V[self.query_indices[row_index].data].data != y[row_index].data[0]][0]
                # print('positive similarity score is {}, and the best negative score is {}'.format(positive_neighbor, negative_neighbor))
                loss += negative_neighbor - positive_neighbor + alpha if (negative_neighbor - positive_neighbor + alpha).data[0] > 0 else 0
                self.index_vector_target_update.append((self.query_indices[row_index,0],normalize(self.K[:, self.query_indices[row_index,0].data[0]] + autograd.Variable(x.data[row_index]), dim=0), None))
                # print('update the K matrix with {} at column {}'.format(self.index_vector_target_update[0][1], self.index_vector_target_update[0][0]))
            else:
                # print('predicted label is wrong.')
                negative_neighbor = self.query_sims[row_index, 0]
                # print('negative similarity score is {}'.format(negative_neighbor))
                if not any((self.V[self.query_indices[row_index].data] == y[row_index].data[0]).data):
                    # print('topk has no correct label. in V there is {}'.format(self.V == y.data[row_index]))
                    positive_neighbors = self.V == y.data[row_index]
                    if not any(positive_neighbors.data):
                        oldest_index = torch.max(self.A, 0)[1].data[0]
                        # oldest_index = random.choice(oldest.data)
                        # print('no positive neighbor found in V. pick the oldest position in A at {}'.format(oldest_index))
                        self.index_vector_target_update.append((oldest_index, autograd.Variable(
                            normalize(self.x.data[row_index], dim=0)), y.data[row_index]))
                        # print('update the K matrix with {} at column {}'.format(self.index_vector_target_update[0][1],
                        #                                                         self.index_vector_target_update[0][0]))

                        continue
                    if torch.numel(torch.nonzero(positive_neighbors.data)) > 1:
                        # print(random.choice(torch.nonzero(positive_neighbors.data)))
                        positive_neighbor = self.K[:, random.choice(torch.nonzero(positive_neighbors.data))[0]]
                        # print('V has multiple correct label, pick one {}'.format(positive_neighbor))
                    else:
                        positive_neighbor = self.K[:,torch.nonzero(positive_neighbors.data)[0,0]]
                        # print('V has only 1 correct label, pick one {}'.format(positive_neighbor))
                    # print(positive_neighbor.size(), self.x[row_index].size())
                    positive_neighbor = torch.dot(positive_neighbor, self.x[row_index])
                    # print('positive similarity is {}'.format(positive_neighbor))
                else:
                    positive_neighbor = self.query_sims[row_index][self.V[self.query_indices[row_index].data] == y[row_index].data[0]][0]
                    # print('topk has the right label. the sim score is {}'.format(positive_neighbor))
                oldest_index = torch.max(self.A, 0)[1].data[0]
                # oldest_index = random.choice(oldest.data)
                # print(oldest_index, row_index)
                self.index_vector_target_update.append((oldest_index, autograd.Variable(normalize(self.x.data[row_index], dim=0)), y.data[row_index]))
                # print('update the K matrix with {} at column {}'.format(self.index_vector_target_update[0][1],
                #                                                         self.index_vector_target_update[0][0]))
                loss += negative_neighbor - positive_neighbor + alpha
                # print('loss is {}'.format(loss))
        # print(loss, accuracy.sum())
        return loss, accuracy

    def update(self):
        for t in self.index_vector_target_update:
            self.K[:, t[0].data[0] if not isinstance(t[0], int) else t[0]] = t[1]
            self.A[t[0].data[0] if not isinstance(t[0], int) else t[0]] = 0
            if t[2] is not None:
                self.V[t[0].data[0] if not isinstance(t[0], int) else t[0]] = t[2]

    def init_K(self, outside_k, outside_v):
        self.K.data[:, :len(outside_k.data)].copy_(outside_k.data.t())
        self.V.data[:len(outside_v.data)].copy_(outside_v.data)
        self.A.data += 1
        self.A.data[:len(outside_v)] = 0

    def cuda(self):
        self.K = self.K.cuda()
        self.V = self.V.cuda()
        self.A = self.A.cuda()


if __name__ == '__main__':

    x = autograd.Variable(torch.Tensor([0.1, 0.1, 0.2]).view(1, -1) / torch.norm(torch.Tensor([0.1, 0.1, 0.2])) )
    x.requires_grad = True
    print(x)
    a = Memory(10, 3)
    a.knn = 5
    a.V[2] = 1
    a.K[:,2] = x.data
    print(a.K)
    outside_K = autograd.Variable(normalize(torch.Tensor([[0.3, 0.1, 0.2],[0.1, 0.1, 0.7]]) ))
    outside_V = autograd.Variable(torch.LongTensor([2, 3]))
    # print(a.query(x))
    a.init_K(outside_K, outside_V)
    y = autograd.Variable(torch.LongTensor([1]))
    loss, _ = a.compute_loss(x,y)
    loss.backward()
    a.update()
    print(a.K, a.V, a.A)

    xp = normalize(torch.rand(4,3))
    print(xp)
    xp = autograd.Variable(xp, requires_grad=True)
    y = autograd.Variable(torch.LongTensor([1, 2, 3, 2]))
    for i, row in enumerate(xp):
        # print(row.view(1, -1))
        # print(a.query(row.view(1, -1)))

        loss, _ = a.compute_loss(row.view(1, -1), y[i])
        print(loss)
        if not isinstance(loss,int):
            loss.backward()
        a.update()
    print(a.K, a.V, a.A)




