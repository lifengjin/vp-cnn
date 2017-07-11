import sys
import os
sys.path.append(os.getcwd())
# print(sys.path)

from bicnn_model import CNN_Mem
from cnn_classifier.model import Memory, CNN_Text
from cnn_classifier.parse_args import parse_args
from cnn_classifier.vpdataset import char_tokenizer,clean_str, vp
from cnn_classifier.train import train
import torch
from torch.autograd import Variable
from torch import nn
from torchtext import data
from bicnn_vpdataset import *
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

# train_iters, dev_iters, test_iters = vp_bicnn(char_field, label_field, args=args,
#                                            num_experts=args.num_experts,
#                                            device=args.device, repeat=False, sort=False, shuffle=False,
#                                            wv_type=args.char_vector, wv_dim=args.char_embed_dim,
#                                            wv_dir=args.char_emb_path,
#                                            min_freq=1)
# train_iter_word, dev_iter_word, test_iter_word = vp_bicnn(word_field, label_field, args=args,
#                                                           num_experts=args.num_experts, device=args.device,
#                                                           repeat=False, sort=False, shuffle=False, wv_type=args.word_vector,
#                                                           wv_dim=args.word_embed_dim, wv_dir=args.emb_path,
#                                                           min_freq=args.min_freq)
xfold = 0


args.cuda = args.yes_cuda and torch.cuda.is_available()
print('cuda is {}'.format(args.cuda))

# print(train_iters)

word_test_results = open('word_test_results.txt', 'w')
char_test_results = open('char_test_results.txt', 'w')

print("\nParameters:", file=log_file_handle)
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value), file=log_file_handle)

for xfold in range(args.xfolds):
    train_iter_word, dev_iter_word, test_iter_word = vp(word_field, label_field, args, foldid=xfold,
                                                        num_experts=args.num_experts, device=args.device,
                                                        repeat=False, sort=False, wv_type=args.word_vector,
                                                        wv_dim=args.word_embed_dim, wv_dir=args.emb_path,
                                                        min_freq=args.min_freq)
    train_iter_char, dev_iter_char, test_iter_char = vp(char_field, label_field, args, foldid=xfold,
                                                        num_experts=args.num_experts, device=args.device,
                                                        repeat=False, sort=False, wv_type=None,
                                                        wv_dim=args.char_embed_dim, wv_dir=None,
                                                        min_freq=args.min_freq)
    args.word_embed_num = len(word_field.vocab.itos)
    args.char_embed_num = len(char_field.vocab.itos)
    print("start pretraining the CNNs")

    labeldata_word_iter = label_iter(word_field)
    labeldata_char_iter = label_iter(char_field)

    print("train the word based models")
    print(word_field.vocab.vectors.size(), type(word_field.vocab.vectors))
    # label_model = CNN_Text(args, 'word', vectors=word_field.vocab.vectors)
    label_model = CNN_Text(args, 'word', vectors=None)
    label_model.cuda()
    args.epochs = 300
    _, label_model = train(labeldata_word_iter, labeldata_word_iter, label_model, args)
    one_iter = labeldata_word_iter.__iter__()
    one_batch = next(one_iter)
    features, labels = one_batch.text, one_batch.label
    hid_label_reps = label_model.confidence(features)
    word_memory = Memory(1000, 359)
    word_memory.init_K(hid_label_reps, labels)
    word_memory.cuda()

    ##test
    word_model = CNN_Mem(args, 'word', vectors=None, mem_size=1000, key_size=359)
    word_model.cnn = label_model
    word_model.memory = word_memory
    args.epochs = args.word_epochs
    bicnn_train.memory_train(train_iter_word, dev_iter_word, word_model, args, log_file_handle=log_file_handle)
    _, test_results = bicnn_train.eval(test_iter_word, word_model, args, log_file_handle=log_file_handle)
    for pred in test_results:
        for ele in pred.data:
            assert isinstance(ele, int) or isinstance(ele, float), type(ele)
            print(ele, file=word_test_results)
    word_test_results.flush()
    ##test

    print('train the char based models')
    label_model = CNN_Text(args, 'char', vectors=None)
    label_model.cuda()
    args.epochs = 30
    _, label_model = train(labeldata_char_iter, labeldata_char_iter, label_model, args)
    one_iter = labeldata_char_iter.__iter__()
    one_batch = next(one_iter)
    features, labels = one_batch.text, one_batch.label
    hid_label_reps = label_model.confidence(features)
    char_memory = Memory(1000, 359)
    char_memory.init_K(hid_label_reps, labels)
    char_memory.cuda()

    ##test
    char_model = CNN_Mem(args, 'char', vectors=None, mem_size=1000, key_size=359)
    char_model.cnn = label_model
    char_model.memory = char_memory
    args.epochs = args.char_epochs
    bicnn_train.memory_train(train_iter_char, dev_iter_char, char_model, args, log_file_handle=log_file_handle)
    _, test_results = bicnn_train.eval(test_iter_char, char_model, args, log_file_handle=log_file_handle)
    for pred in test_results:
        for ele in pred.data:
            assert isinstance(ele, int) or isinstance(ele, float), type(ele)
            print(ele, file=char_test_results)
    char_test_results.flush()
    ##test

    # log_file_handle.write('Fold {}, word\n'.format(xfold))
    # train_iter = train_iter_word[xfold]
    # dev_iter = dev_iter_word[xfold]
    # # test_iter = test_iter_word[xfold]
    # word_model = CNN_Mem(args, 'word', vectors=None).load_state_dict(word_model)
    # word_model.load_state_dict(word_model_dict)
    # bicnn_train.train(train_iter, dev_iter, word_model, args, log_file_handle=log_file_handle)

    log_file_handle.flush()
