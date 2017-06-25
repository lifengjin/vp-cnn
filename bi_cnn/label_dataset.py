from torchtext import data
import os
import math, random
import pickle

def label_iter(text_field):
    dataset = Label_dataset(text_field)
    return data.Iterator(dataset, batch_size=256, shuffle=True)

class Label_dataset(data.Dataset):

    label_file = 'data/labels.txt'

    def __init__(self, text_field):
        label_field = data.Field(sequential=False, use_vocab=False)
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
                        [labels[label], labels[label2], '1' if label == label2 else '-1'], fields)
                    examples += [this_example]

        super(Label_dataset, self).__init__(examples, fields)