import sys
import os
sys.path.append(os.getcwd())
# print(sys.path)

from bicnn_model import bi_CNN_Text
import torch
from cnn_classifier.parse_args import parse_args
from torch.autograd import Variable
from torch import nn
from cnn_classifier.vpdataset import char_tokenizer
from torchtext import data
from bicnn_vpdataset import *
from cnn_classifier.vpdataset import clean_str
import bicnn_train
from label_dataset import label_iter


args = parse_args()
prediction_file_handle = open(args.prediction_file_handle, 'w')
log_file_handle = open(args.log_file, 'w')

print("\nLoading data...")

tokenizer = data.Pipeline(clean_str)
char_field = data.Field(lower=True, tokenize=char_tokenizer)
word_field = data.Field(lower=True, tokenize=tokenizer)
label_field = data.Field(sequential=False, use_vocab=False, preprocessing=int)

train_iters, dev_iters, test_iters = vp_bicnn(char_field, label_field, args=args,
                                           num_experts=args.num_experts,
                                           device=args.device, repeat=False, sort=False, shuffle=False,
                                           wv_type=args.char_vector, wv_dim=args.char_embed_dim,
                                           wv_dir=args.char_emb_path,
                                           min_freq=1)
train_iter_word, dev_iter_word, test_iter_word = vp_bicnn(word_field, label_field, args=args,
                                                          num_experts=args.num_experts, device=args.device,
                                                          repeat=False, sort=False, shuffle=False, wv_type=args.word_vector,
                                                          wv_dim=args.word_embed_dim, wv_dir=args.emb_path,
                                                          min_freq=args.min_freq)
args.cuda = args.yes_cuda and torch.cuda.is_available()
print('cuda is {}'.format(args.cuda))
args.word_embed_num = len(word_field.vocab.itos)
args.char_embed_num = len(char_field.vocab.itos)
# print(train_iters)

print("start pretraining the CNNs")

labeldata_word_iter = label_iter(word_field)
labeldata_char_iter = label_iter(char_field, multiplier=5)

word_model = bi_CNN_Text(args, 'word', vectors=word_field.vocab.vectors)
bicnn_train.train_label(word_model, labeldata_word_iter, args)
word_model_dict = word_model.state_dict()
word_model = None
labeldata_word_iter = None

char_model = bi_CNN_Text(args, 'char')
bicnn_train.train_label(char_model, labeldata_char_iter, args)
char_model_dict = char_model.state_dict()
char_model = None
labeldata_char_iter = None

for xfold in range(args.xfolds):
    # if xfold != 9:
    #     continue
    log_file_handle.write('Fold {}, char\n'.format(xfold))
    train_iter = train_iters[xfold]
    dev_iter = dev_iters[xfold]
    # test_iter = test_iters[xfold]

    char_model = bi_CNN_Text(args, 'char', vectors=None)
    char_model.load_state_dict(char_model_dict)
    bicnn_train.train(train_iter, dev_iter, char_model, args, log_file_handle=log_file_handle)

    log_file_handle.write('Fold {}, word\n'.format(xfold))
    train_iter = train_iter_word[xfold]
    dev_iter = dev_iter_word[xfold]
    # test_iter = test_iter_word[xfold]
    word_model = bi_CNN_Text(args, 'word', vectors=None).load_state_dict(word_model)
    word_model.load_state_dict(word_model_dict)
    bicnn_train.train(train_iter, dev_iter, word_model, args, log_file_handle=log_file_handle)
    # print("\nParameters:", file=log_file_handle)
    # for attr, value in sorted(args.__dict__.items()):
    #     print("\t{}={}".format(attr.upper(), value), file=log_file_handle)
    log_file_handle.flush()
