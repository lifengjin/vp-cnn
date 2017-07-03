from torchtext import data
import os
import math, random
import pickle
import torch

def label_iter(text_field):
    dataset = Label_dataset(text_field)
    return data.Iterator(dataset, batch_size=len(dataset.examples), shuffle=False, repeat=False)

def bi_label_iter(text_field, multiplier = 30):
    dataset = bi_Label_dataset(text_field)
    return data.Iterator(dataset, batch_size=359*multiplier, shuffle=False, repeat=False)

class bi_Label_dataset(data.Dataset):

    label_file = '/home/jin.544/mulproj/vp-cnn/data/labels.txt'

    def __init__(self, text_field):
        label_field = data.Field(sequential=False, use_vocab=False, preprocessing=float, tensor_type=torch.FloatTensor)
        fields = [("s1", text_field), ("s2", text_field), ('target', label_field)]
        labels = {}
        examples = []
        with open(self.label_file) as f:
            lines = f.readlines()
            # pdb.set_trace()
            for line in lines:
                label, text = line.split("\t")
                labels[label] = text
            for label in labels.keys():
                for label2 in labels.keys():
                    this_example = data.Example.fromlist(
                        [labels[label], labels[label2], '1' if label == label2 else '0'], fields)
                    examples += [this_example]

        super(bi_Label_dataset, self).__init__(examples, fields)

class Label_dataset(data.Dataset):

    label_file = '/home/jin.544/mulproj/vp-cnn/data/labels.txt'

    def __init__(self, text_field):
        label_field = data.Field(sequential=False, use_vocab=False, preprocessing=int, tensor_type=torch.LongTensor)
        fields = [("text", text_field), ('label', label_field)]
        examples = []
        with open(self.label_file) as f:
            lines = f.readlines()
            # pdb.set_trace()
            for line in lines:
                label, text = line.split("\t")
                this_example = data.Example.fromlist(
                        [text, label], fields)
                examples += [this_example]

        super(Label_dataset, self).__init__(examples, fields)