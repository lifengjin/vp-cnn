import torch
import torch.nn.functional as F
import sys
import copy
from tqdm import *

def memory_train(train_iter, dev_iter, model, args, **kwargs):
    if args.cuda:
        model.cuda()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.cnn.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.cnn.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.cnn.parameters(), rho=0.95)
    else:
        raise Exception("bad optimizer!")

    steps = 0
    model.train()
    best_acc = 0
    best_model = None
    print('training the memory CNN starts now.')
    for epoch in range(1, args.epochs+1):
        corrects = 0
        total_loss = 0
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(0)  # batch first, index align
            # print(feature)
            # print(train_iter.data().fields['text'].vocab.stoi)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            assert feature.volatile is False and target.volatile is False
            # print(feature, target)
            optimizer.zero_grad()
            # print(feature ,target)
            loss, accuracy = model(feature, target)
            if loss.data[0] != 0:
                loss.backward()
                total_loss += loss
                optimizer.step()
            model.update_mem()

            # # max norm constraint
            # if args.max_norm > 0:
            #     if not args.no_always_norm:
            #         for row in model.fc1.weight.data:
            #             norm = row.norm() + 1e-7
            #             row.div_(norm).mul_(args.max_norm)
            #     else:
            #         model.fc1.weight.data.renorm_(2, 0, args.max_norm)

            corrects += accuracy.sum().float()
        sys.stdout.write(
                'Epoch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(epoch,
                                                                         total_loss.data[0],
                                                                         corrects.data[0] / len(train_iter.dataset),
                                                                         corrects.data[0], len(train_iter.dataset)
                                                                         ))
            # if steps % args.test_interval == 0:
            #     if args.verbose:
            #         corrects = accuracy.sum()
            #         accuracy = corrects/batch.batch_size * 100.0
            #         print(
            #         'Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
            #                                                                  loss.data[0],
            #                                                                  accuracy.data[0],
            #                                                                  str(corrects),
            #                                                                  batch.batch_size), file=kwargs['log_file_handle'])
        acc = eval(dev_iter, model, args, **kwargs)
    #     if acc > best_acc:
    #         best_acc = acc
    #         best_model = copy.deepcopy(model)
    # model = best_model
    # acc = eval(dev_iter, model, args, **kwargs)
    return acc, model

def eval(data_iter, model, args, **kwargs):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(0)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        _, accuracy = model(feature, target, update=False)

        corrects += accuracy.sum()

    size = len(data_iter.dataset)
    accuracy = corrects.float()/size * 100.0
    model.train()
    print('\nEvaluation - acc: {:.4f}%({}/{})'.format(
                                                                       accuracy.data[0],
                                                                       corrects.data[0],
                                                                       size))
    if args.verbose:
        print('Evaluation - acc: {:.4f}%({}/{})'.format(
                                                                           accuracy,
                                                                           corrects.data[0],
                                                                           size), file=kwargs['log_file_handle'])
    return accuracy

def bi_train(train_iter, dev_iter, model, args, **kwargs):
    if args.cuda:
        model.cuda()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95)
    else:
        raise Exception("bad optimizer!")
    multiplier_batch = int(args.batch_size / 359)
    steps = 0
    model.train()
    best_acc = 0
    best_model = None
    # objf = torch.nn.CosineEmbeddingLoss()
    dev_loss = 0
    train_loss = 0
    for epoch in range(1, args.epochs + 1):
        i = -1
        acc = 0
        for batch in train_iter:
            i += 1
            s1, s2, target = batch.s1, batch.s2, batch.label
            assert torch.sum(batch.label.data) == -357 * multiplier_batch, str(torch.sum(batch.label.data))
            s1.data.t_(), s2.data.t_()  # batch first, index align
            target = target.view(multiplier_batch, -1)
            val, index = torch.max(target, 1)
            assert torch.sum(val) == multiplier_batch
            # print(index)

            assert s1.volatile is False and index.volatile is False
            # print(feature, target)
            optimizer.zero_grad()
            sim_score = model.compute_similarity((s1, s2))

            # loss = objf(sim_score[0], sim_score[1], target)
            y = sim_score.view(multiplier_batch, -1)
            y = F.log_softmax(y)
            print(y.size())
            loss = F.nll_loss(y, index)
            loss.backward()
            for param in model.parameters():
                param.grad.data = param.grad.data / 359
            _, y_index = torch.max(y, 1)
            if torch.equal(y_index.data, index.data):
                local_acc = 1
            else:
                local_acc = 0
            acc += local_acc
            optimizer.step()
            train_loss += loss.data[0]
            sys.stdout.write(
                '\rEpoch {} Sample {} - loss: {:.6f} local acc: {} target {} predicted {})'.format(epoch, i,
                                                                                                   loss.data[0],
                                                                                                   local_acc
                                                                                                   , index.data[0],
                                                                                                   y_index.data[0]))
            # max norm constraint
            if args.max_norm > 0:
                if not args.no_always_norm:
                    for row in model.cnn.fc1.weight.data:
                        norm = row.norm() + 1e-7
                        row.div_(norm).mul_(args.max_norm)
                else:
                    model.cnn.fc1.weight.data.renorm_(2, 0, args.max_norm)

            steps += 1
        sys.stdout.write(
                '\rEpoch {} - loss: {:.6f} acc: {} batch: {})'.format(epoch, train_loss / len(train_iter.dataset),
                                                                      acc / (i + 1), args.batch_size))
        train_loss = 0
            # if steps % args.test_interval == 0:
            #     if args.verbose:
            #         corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            #         accuracy = corrects/batch.batch_size * 100.0
            #         print(
            #         'Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
            #                                                                  loss.data[0],
            #                                                                  accuracy,
            #                                                                  corrects,
            #                                                                  batch.batch_size), file=kwargs['log_file_handle'])
        acc = eval(dev_iter, model, args, **kwargs)
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)
            # print(model.embed.weight[100])
            # if steps % args.save_interval == 0:
            #     if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
            #     save_prefix = os.path.join(args.save_dir, 'snapshot')
            #     save_path = '{}_steps{}.pt'.format(save_prefix, steps)
            #     torch.save(model, save_path)
    model = best_model
    acc = eval(dev_iter, model, args, **kwargs)
    return acc, model


def bi_eval(data_iter, model, args, **kwargs):
    model.eval()
    total_loss = 0
    acc = 0
    i = -1
    for batch in data_iter:
        i += 1
        s1, s2, target = batch.s1, batch.s2, batch.label
        s1.data.t_(), s2.data.t_()  # batch first, index align
        _, index = torch.max(target, 0)

        assert s1.volatile is True
        # print(feature, target)
        sim_score = model((s1, s2))
        y = F.cosine_similarity(sim_score[0], sim_score[1]).view(1, -1)
        y = F.log_softmax(y)
        _, y_index = torch.max(y, 1)
        if torch.equal(y_index.data, index.data):
            local_acc = 1
        else:
            local_acc = 0
        acc += local_acc

    size = i + 1
    acc = acc / size
    model.train()
    print('\nEvaluation - acc: {:.6f} )'.format(acc))
    if args.verbose:
        print('Evaluation - acc: {:.6f} )'.format(acc), file=kwargs['log_file_handle'])
    return acc

def bi_train_label(model, train_iter, args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95)
    model.train()
    objf = torch.nn.MSELoss()
    for epoch in range(1, args.epochs + 1):
        train_loss = 0
        this_epoch_iter = train_iter.__iter__()
        print("epoch {} for pretraining".format(epoch))
        for i in trange(len(train_iter)):
            batch = next(this_epoch_iter)
            s1, s2, target = batch.s1, batch.s2, batch.target
            s1.data.t_(), s2.data.t_()  # batch first, index align
            # print(s1.size(), s2.size())
            assert s1.volatile is False and target.volatile is False
            assert isinstance(s1.data, torch.cuda.LongTensor), type(s1.data)
            optimizer.zero_grad()
            sim_score = model.compute_similarity((s1, s2))
            # print(sim_score.size(), target.size())
            # print(sim_score)
            loss = objf(sim_score, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
        sys.stdout.write(
                'Epoch {} - loss: {:.6f})'.format(epoch, train_loss / len(train_iter.dataset)))
    else:
        _, hid_labels = model.forward((s1, s2))
        hid_labels = hid_labels[-359:]
    return model, hid_labels