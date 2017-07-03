from torchtext import data
import os
import pdb
import random
import math
import re
import torch

def vp(text_field, label_field, args, foldid, num_experts=0, **kargs):
    # print('num_experts', num_experts)
    train_data, dev_data, test_data = VP.splits(text_field, label_field, foldid=foldid,
                                                          num_experts=num_experts)
    if num_experts > 0:
        text_field.build_vocab(train_data[0], dev_data[0], test_data, wv_type=kargs["wv_type"], wv_dim=kargs["wv_dim"],
                               wv_dir=kargs["wv_dir"], min_freq=kargs['min_freq'])
    else:
        text_field.build_vocab(train_data, dev_data, test_data, wv_type=kargs["wv_type"], wv_dim=kargs["wv_dim"],
                               wv_dir=kargs["wv_dir"], min_freq=kargs['min_freq'])
    # label_field.build_vocab(train_data, dev_data, test_data)
    kargs.pop('wv_type')
    kargs.pop('wv_dim')
    kargs.pop('wv_dir')
    kargs.pop("min_freq")
    # print(type(train_data), type(dev_data))
    if num_experts > 0:
        train_iter = []
        dev_iter = []
        for i in range(num_experts):
            this_train_iter, this_dev_iter, test_iter = data.Iterator.splits((train_data[i], dev_data[i], test_data),
                                                                             batch_sizes=(args.batch_size,
                                                                                          len(dev_data[i]),
                                                                                          len(test_data)), **kargs)
            train_iter.append(this_train_iter)
            dev_iter.append(this_dev_iter)
    else:
        train_iter, dev_iter, test_iter = data.Iterator.splits(
            (train_data, dev_data, test_data),
            batch_sizes=(args.batch_size,
                         len(dev_data),
                         len(test_data)),
            **kargs)
    return train_iter, dev_iter, test_iter

class VP(data.Dataset):
    """modeled after Shawn1993 github user's Pytorch implementation of Kim2014 - cnn for text categorization"""

    filename = "/home/jin.544/vp-cnn/data/wilkins_corrected.shuffled.51.txt"

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        """Create a virtual patient (VP) dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        #no preprocessing needed 
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
                path = self.dirname if path is None else path
                examples = []
                with open(os.path.join(path, self.filename)) as f:
                    lines = f.readlines()
                    #pdb.set_trace()
                    for line in lines:
                        label, text = line.split("\t")
                        this_example = data.Example.fromlist([text, label], fields)
                        examples += [this_example]

                    #assume "target \t source", one instance per line
        # print(examples[0].text)
        super(VP, self).__init__(examples, fields, **kwargs)
        

    @classmethod
    #def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True ,root='.', **kwargs):
    def splits(cls, text_field, label_field, numfolds=10, foldid=None, dev_ratio=.1, shuffle=False, root='.',
               num_experts=0, **kwargs):
        
        """Create dataset objects for splits of the VP dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        #path = cls.download_or_unzip(root)
        #examples = cls(text_field, label_field, path=path, **kwargs).examples
        examples = cls(text_field, label_field, path=root, **kwargs).examples
        if shuffle: random.shuffle(examples)
        fields = [('text', text_field), ('label', label_field)]
        label_examples = []
        label_filename = '/home/jin.544/vp-cnn/data/labels.txt'
        with open(label_filename) as f:
            lines = f.readlines()
            # pdb.set_trace()
            for line in lines:
                label, text = line.split("\t")
                this_example = data.Example.fromlist([text, label], fields)
                label_examples += [this_example]
        
        if foldid==None:
            dev_index = -1 * int(dev_ratio*len(examples))
            return (cls(text_field, label_field, examples=examples[:dev_index]),
                    cls(text_field, label_field, examples=examples[dev_index:]))
        else:
            #get all folds
            fold_size = math.ceil(len(examples)/numfolds)
            folds = []
            for fold in range(numfolds):
                startidx = fold*fold_size
                endidx = startidx+fold_size if startidx+fold_size < len(examples) else len(examples)
                folds += [examples[startidx:endidx]]

            #take all folds except foldid as training/dev
            traindev = [fold for idx, fold in enumerate(folds) if idx != foldid]
            traindev = [item for sublist in traindev for item in sublist]
            dev_index = -1 * int(dev_ratio*len(traindev))

            #test will be entire held out section (foldid)
            test = folds[foldid]
            # print(len(traindev[:dev_index]), 'num_experts', num_experts)
            if num_experts > 0:
                assert num_experts <= 5
                trains = []
                devs = []
                dev_length = math.floor(len(traindev) * dev_ratio)
                # print(dev_length)
                for i in range(num_experts):
                    devs.append(cls(text_field, label_field, examples=traindev[dev_length*i:dev_length*(i+1)]))
                    trains.append(cls(text_field, label_field, examples=traindev[:dev_length*i]+traindev[dev_length*(i+1):]+label_examples))
                return (trains, devs, cls(text_field, label_field, examples=test))

            else:
                return (cls(text_field, label_field, examples=traindev[:dev_index]+label_examples),
                    cls(text_field, label_field, examples=traindev[dev_index:]),
                    cls(text_field, label_field, examples=test))

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub("[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub("\'s", " \'s", string)
  string = re.sub("\'m", " \'m", string)
  string = re.sub("\'ve", " \'ve", string)
  string = re.sub("n\'t", " n\'t", string)
  string = re.sub("\'re", " \'re", string)
  string = re.sub("\'d", " \'d", string)
  string = re.sub("\'ll", " \'ll", string)
  string = re.sub(",", " , ", string)
  string = re.sub("!", " ! ", string)
  string = re.sub("\(", " ( ", string)
  string = re.sub("\)", " ) ", string)
  string = re.sub("\?", " ? ", string)
  string = re.sub("\s{2,}", " ", string)
  return pad2(string.strip().lower().split(" "))

def pad2(x):
    x = ['<pad>', '<pad>', '<pad>', '<pad>'] + x
    return x


def char_tokenizer(mstring):
    return ['<pad>']*5 + list(mstring)
