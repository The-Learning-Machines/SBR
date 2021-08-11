import argparse
import pickle
import time
import sys

from proc_utils import Dataset, split_validation
from model import *
from torch.utils.tensorboard import SummaryWriter


def str2bool(v):
    return v.lower() in ('true')

# Default args used for Diginetica 

class Diginetica_arg():
    dataset = 'diginetica'
    batchSize = 50
    hiddenSize = 100
    epoch = 30
    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 3
    l2 = 1e-5
    step = 1
    patience = 10
    nonhybrid = True
    validation = True
    valid_portion = 0.1


# Default args used for Yoochoose1_64

class Yoochoose_arg():
    dataset = 'yoochoose1_64'
    batchSize = 75
    hiddenSize = 120
    epoch = 30
    lr = 0.001
    lr_dc = 0.1
    lr_dc_step = 3
    l2 = 1e-5
    step = 1
    patience = 10
    nonhybrid = True
    validation = True
    valid_portion = 0.1


def main(opt):
    model_save_dir = 'saved/'
    writer = SummaryWriter(log_dir='with_pos/logs')

    if opt.dataset == 'diginetica':
        train_data = pickle.load(
            open('datasets/cikm16/raw' + '/train.txt', 'rb'))
        test_data = pickle.load(
            open('datasets/cikm16/raw' + '/test.txt', 'rb'))

    elif opt.dataset == 'yoochoose1_64':
        train_data = pickle.load(
            open('datasets/yoochoose1_64/raw' + '/train.txt', 'rb'))
        test_data = pickle.load(
            open('datasets/yoochoose1_64/raw' + '/test.txt', 'rb'))

    if opt.validation:
        train_data, valid_data = split_validation(
            train_data, opt.valid_portion)
        test_data = valid_data
    else:
        print('Testing dataset used validation set')

    train_data = Dataset(train_data, shuffle=True)
    test_data = Dataset(test_data, shuffle=False)

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    model = to_cuda(Attention_SessionGraph(opt, n_node))
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-' * 50)
        print('Epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0

        # Logging
        writer.add_scalar('epoch/recall', hit, epoch)
        writer.add_scalar('epoch/mrr', mrr, epoch)

        flag = 0

        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
            torch.save(model, model_save_dir + 'epoch_' +
                       str(epoch) + '_recall_' + str(hit) + '_.pt')
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
            torch.save(model, model_save_dir + 'epoch_' +
                       str(epoch) + '_mrr_' + str(mrr) + '_.pt')

        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' %
              (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))

        bad_counter += 1 - flag

        if bad_counter >= opt.patience:
            break

    print('-' * 50)
    end = time.time()
    print("Running time: %f seconds" % (end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='diginetica',
                        help='Dataset name: diginetica | yoochoose1_64')
    parser.add_argument('--defaults', type=str2bool,
                        default=True, help='Use default configuration')
    parser.add_argument('--batchSize', type=int,
                        default=50, help='Batch size')
    parser.add_argument('--hiddenSize', type=int,
                        default=100, help='Hidden state dimensions')
    parser.add_argument('--epoch', type=int, default=30,
                        help='The number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Set the Learning Rate')
    parser.add_argument('--lr_dc', type=float, default=0.1,
                        help='Set the decay rate for Learning rate')
    parser.add_argument('--lr_dc_step', type=int, default=3,
                        help='Steps for learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5,
                        help='Assign L2 Penalty')
    parser.add_argument('--patience', type=int, default=10,
                        help='Used for early stopping criterion')
    parser.add_argument('--validation', action='store_true', help='validation')
    parser.add_argument('--valid_portion', type=float, default=0.1,
                        help='Portion of train set to split into val set')
    opt = parser.parse_args()

    if opt.defaults:
        if opt.dataset == 'diginetica':
            opt = Diginetica_arg()

        else:
            opt = Yoochoose_arg()

    else:
        print("Not using the default configuration")

    main(opt)
