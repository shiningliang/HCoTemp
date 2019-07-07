import ujson as json
import pickle as pkl
import numpy as np
import logging
import os
import argparse
import random
import torch
import torch.optim as optim
from rec_model import COTEMP
from rec_preprocess import run_prepare
from rec_util import train_one_epoch, valid_batch

# records = [{"user": [[1, 3, 5], [2, 3, 4]], "item": [[2, 3, 4], [1, 2, 5]]},
#            {"user": [[1, 2, 4], [3, 4, 5]], "item": [[2, 4, 5], [1, 2, 3]]}]
#
# with open(demo_name, 'w') as f:
#     for i in range(len(records)):
#         f.write(json.dumps(records[i]) + '\n')
# f.close()
#

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Rec')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train and valid the model')
    parser.add_argument('--test', action='store_true',
                        help='evaluate the model on test set')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--seed', type=int, default=23333,
                        help='random seed (default: 23333)')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--disable_cuda', action='store_true',
                                help='Disable CUDA')
    train_settings.add_argument('--lr', type=float, default=0.0001,
                                help='learning rate')
    train_settings.add_argument('--clip', type=float, default=0.35,
                                help='gradient clip, -1 means no clip (default: 0.35)')
    train_settings.add_argument('--weight_decay', type=float, default=0.0003,
                                help='weight decay')
    train_settings.add_argument('--emb_dropout', type=float, default=0.5,
                                help='dropout keep rate')
    train_settings.add_argument('--layer_dropout', type=float, default=0.5,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_train', type=int, default=32,
                                help='train batch size')
    train_settings.add_argument('--batch_eval', type=int, default=32,
                                help='dev batch size')
    train_settings.add_argument('--epochs', type=int, default=50,
                                help='train epochs')
    train_settings.add_argument('--optim', default='Adam',
                                help='optimizer type')
    train_settings.add_argument('--patience', type=int, default=2,
                                help='num of epochs for train patients')
    train_settings.add_argument('--period', type=int, default=50,
                                help='period to save batch loss')
    train_settings.add_argument('--num_threads', type=int, default=8,
                                help='Number of threads in input pipeline')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--max_len', type=int, default=3,
                                help='max record length in a year')
    model_settings.add_argument('--T', type=int, default=20,
                                help='length of the year sequence')
    model_settings.add_argument('--NU', type=int, default=936,
                                help='num of users')
    model_settings.add_argument('--NI', type=int, default=2049,
                                help='num of items')
    model_settings.add_argument('--NF', type=int, default=32,
                                help='num of factors')
    model_settings.add_argument('--n_hidden', type=int, default=64,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--n_layer', type=int, default=2,
                                help='num of layers')
    model_settings.add_argument('--is_atten', type=bool, default=False,
                                help='whether to use self attention')
    model_settings.add_argument('--n_block', type=int, default=4,
                                help='attention block size (default: 2)')
    model_settings.add_argument('--n_head', type=int, default=4,
                                help='attention head size (default: 2)')
    model_settings.add_argument('--is_pos', type=bool, default=False,
                                help='whether to use position embedding')
    model_settings.add_argument('--is_sinusoid', type=bool, default=True,
                                help='whether to use sinusoid position embedding')
    model_settings.add_argument('--n_kernel', type=int, default=3,
                                help='kernel size (default: 3)')
    model_settings.add_argument('--n_kernels', type=int, default=[2, 3, 4],
                                help='kernels size (default: 2, 3, 4)')
    model_settings.add_argument('--n_level', type=int, default=6,
                                help='# of levels (default: 10)')
    model_settings.add_argument('--n_filter', type=int, default=50,
                                help='number of hidden units per layer (default: 256)')
    model_settings.add_argument('--n_class', type=int, default=2,
                                help='class size (default: 2)')
    model_settings.add_argument('--kmax_pooling', type=int, default=2,
                                help='top-K max pooling')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--task', default='dvd',
                               help='the task name')
    path_settings.add_argument('--model', default='COTEMP',
                               help='the model name')
    path_settings.add_argument('--user_record_file', default='user_record.json',
                               help='the record file name')
    path_settings.add_argument('--item_record_file', default='item_record.json',
                               help='the record file name')
    path_settings.add_argument('--train_file', default='train.csv',
                               help='the train file name')
    path_settings.add_argument('--valid_file', default='dev.csv',
                               help='the valid file name')
    path_settings.add_argument('--test_file', default='test.csv',
                               help='the test file name')
    path_settings.add_argument('--raw_dir', default='data/raw_data/',
                               help='the dir to store raw data')
    path_settings.add_argument('--processed_dir', default='data/processed_data/',
                               help='the dir to store prepared data')
    path_settings.add_argument('--outputs_dir', default='outputs/',
                               help='the dir for outputs')
    path_settings.add_argument('--model_dir', default='models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='results/',
                               help='the dir to store the results')
    path_settings.add_argument('--pics_dir', default='pics/',
                               help='the dir to store the pictures')
    path_settings.add_argument('--summary_dir', default='summary/',
                               help='the dir to write tensorboard summary')
    return parser.parse_args()


def train(args, file_paths):
    logger = logging.getLogger('Rec')
    logger.info('Loading train file...')
    with open(file_paths.train_file, 'rb') as fh:
        train_file = pkl.load(fh)
    fh.close()
    logger.info('Loading valid file...')
    with open(file_paths.valid_file, 'rb') as fh:
        valid_file = pkl.load(fh)
    logger.info('Loading record file...')
    with open(file_paths.user_record_file, 'rb') as fh:
        user_record_file = pkl.load(fh)
    fh.close()
    with open(file_paths.item_record_file, 'rb') as fh:
        item_record_file = pkl.load(fh)
    fh.close()

    train_num = len(train_file['labels'])
    valid_num = len(valid_file['labels'])
    logger.info('Num of train data {} valid data {}'.format(train_num, valid_num))
    user_num = len(user_record_file)
    item_num = len(item_record_file)
    logger.info('Num of users {} items {}'.format(user_num, item_num))

    logger.info('Initialize the model...')
    UEM = np.random.normal(0., 0.01, (args.T * args.NU + 1, args.NF))
    UEM[0] = 0.
    IEM = np.random.normal(0., 0.01, (args.T * args.NI + 1, args.NF))
    IEM[0] = 0.
    dropout = {'emb': args.emb_dropout, 'layer': args.layer_dropout}
    model = COTEMP(UEM, IEM, args.T, args.NU, args.NI, args.NF, args.n_class, args.n_hidden, args.n_layer, dropout,
                   logger).to(device=args.device)
    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, patience=args.patience, verbose=True)

    max_acc, max_p, max_r, max_f, max_sum, max_epoch = 0, 0, 0, 0, 0, 0
    FALSE = {}
    ROC = {}
    PRC = {}
    for ep in range(1, args.epochs + 1):
        logger.info('Training the model for epoch {}'.format(ep))
        avg_loss = train_one_epoch(model, optimizer, train_num, train_file, user_record_file, item_record_file, args,
                                   logger)
        logger.info('Epoch {} AvgLoss {}'.format(ep, avg_loss))

        logger.info('Evaluating the model for epoch {}'.format(ep))
        eval_metrics, fpr, tpr, precision, recall = valid_batch(model, valid_num, args.batch_eval, valid_file,
                                                                user_record_file, item_record_file, args.device,
                                                                'valid', logger)
        logger.info('Valid Loss - {}'.format(eval_metrics['loss']))
        logger.info('Valid Acc - {}'.format(eval_metrics['acc']))
        logger.info('Valid Precision - {}'.format(eval_metrics['precision']))
        logger.info('Valid Recall - {}'.format(eval_metrics['recall']))
        logger.info('Valid F1 - {}'.format(eval_metrics['f1']))
        logger.info('Valid AUCROC - {}'.format(eval_metrics['auc_roc']))
        logger.info('Valid AUCPRC - {}'.format(eval_metrics['auc_prc']))
        max_acc = max((eval_metrics['acc'], max_acc))
        max_p = max(eval_metrics['precision'], max_p)
        max_r = max(eval_metrics['recall'], max_r)
        max_f = max(eval_metrics['f1'], max_f)
        # valid_sum = eval_metrics['precision'] + eval_metrics['recall'] + eval_metrics['f1']
        valid_sum = eval_metrics['auc_roc'] + eval_metrics['auc_prc']
        if valid_sum > max_sum:
            max_sum = valid_sum
            max_epoch = ep
            FALSE = {'FP': eval_metrics['fp'], 'FN': eval_metrics['fn']}
            ROC = {'FPR': fpr, 'TPR': tpr}
            PRC = {'PRECISION': precision, 'RECALL': recall}
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.bin'))

        scheduler.step(metrics=eval_metrics['f1'])

        randnum = random.randint(0, 1e8)
        random.seed(randnum)
        random.shuffle(train_file['uids'])
        random.seed(randnum)
        random.shuffle(train_file['iids'])
        random.seed(randnum)
        random.shuffle(train_file['labels'])

    logger.info('Max Acc - {}'.format(max_acc))
    logger.info('Max Precision - {}'.format(max_p))
    logger.info('Max Recall - {}'.format(max_r))
    logger.info('Max F1 - {}'.format(max_f))
    logger.info('Max Epoch - {}'.format(max_epoch))
    logger.info('Max Sum - {}'.format(max_sum))
    with open(os.path.join(args.result_dir, 'FALSE_valid.json'), 'w') as f:
        f.write(json.dumps(FALSE) + '\n')
    f.close()
    with open(os.path.join(args.result_dir, 'ROC_valid.json'), 'w') as f:
        f.write(json.dumps(ROC) + '\n')
    f.close()
    with open(os.path.join(args.result_dir, 'PRC_valid.json'), 'w') as f:
        f.write(json.dumps(PRC) + '\n')
    f.close()


def test(args, file_paths):
    logger = logging.getLogger('Rec')
    logger.info('Loading test file...')
    with open(file_paths.test_file, 'rb') as fh:
        test_file = pkl.load(fh)
    fh.close()
    logger.info('Loading record file...')
    with open(file_paths.user_record_file, 'rb') as fh:
        user_record_file = pkl.load(fh)
    fh.close()
    with open(file_paths.item_record_file, 'rb') as fh:
        item_record_file = pkl.load(fh)
    fh.close()

    test_num = len(test_file['labels'])
    logger.info('Num of test data {}'.format(test_num))
    user_num = len(user_record_file)
    item_num = len(item_record_file)
    logger.info('Num of users {} items {}'.format(user_num, item_num))

    logger.info('Initialize the model...')
    UEM = np.random.normal(0., 0.01, (args.T * args.NU + 1, args.NF))
    UEM[0] = 0.
    IEM = np.random.normal(0., 0.01, (args.T * args.NI + 1, args.NF))
    IEM[0] = 0.
    dropout = {'emb': args.emb_dropout, 'layer': args.layer_dropout}
    model = COTEMP(UEM, IEM, args.T, args.NU, args.NI, args.NF, args.n_class, args.n_hidden, args.n_layer, dropout,
                   logger).to(device=args.device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.bin')))

    eval_metrics, fpr, tpr, precision, recall = valid_batch(model, test_num, args.batch_eval, test_file,
                                                            user_record_file, item_record_file, args.device,
                                                            'test', logger)
    logger.info('Test Loss - {}'.format(eval_metrics['loss']))
    logger.info('Test Acc - {}'.format(eval_metrics['acc']))
    logger.info('Test Precision - {}'.format(eval_metrics['precision']))
    logger.info('Test Recall - {}'.format(eval_metrics['recall']))
    logger.info('Test F1 - {}'.format(eval_metrics['f1']))
    logger.info('Test AUCROC - {}'.format(eval_metrics['auc_roc']))
    logger.info('Test AUCPRC - {}'.format(eval_metrics['auc_prc']))

    FALSE = {'FP': eval_metrics['fp'], 'FN': eval_metrics['fn']}
    ROC = {'FPR': fpr, 'TPR': tpr}
    PRC = {'PRECISION': precision, 'RECALL': recall}

    with open(os.path.join(args.result_dir, 'FALSE_test.json'), 'w') as f:
        f.write(json.dumps(FALSE) + '\n')
    f.close()
    with open(os.path.join(args.result_dir, 'ROC_test.json'), 'w') as f:
        f.write(json.dumps(ROC) + '\n')
    f.close()
    with open(os.path.join(args.result_dir, 'PRC_test.json'), 'w') as f:
        f.write(json.dumps(PRC) + '\n')
    f.close()


if __name__ == '__main__':
    args = parse_args()

    logger = logging.getLogger('Rec')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info('Running with args : {}'.format(args))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    logger.info('Preparing the directories...')
    args.processed_dir = os.path.join(args.processed_dir, args.task)
    args.model_dir = os.path.join(args.outputs_dir, args.task, args.model, args.model_dir)
    args.result_dir = os.path.join(args.outputs_dir, args.task, args.model, args.result_dir)
    args.pics_dir = os.path.join(args.outputs_dir, args.task, args.model, args.pics_dir)
    args.summary_dir = os.path.join(args.outputs_dir, args.task, args.model, args.summary_dir)
    for dir_path in [args.raw_dir, args.processed_dir, args.model_dir, args.result_dir, args.pics_dir,
                     args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    class FilePaths(object):
        def __init__(self):
            self.train_file = os.path.join(args.processed_dir, 'train.pkl')
            self.valid_file = os.path.join(args.processed_dir, 'valid.pkl')
            self.test_file = os.path.join(args.processed_dir, 'test.pkl')
            self.user_record_file = os.path.join(args.processed_dir, 'user_record.pkl')
            self.item_record_file = os.path.join(args.processed_dir, 'item_record.pkl')


    file_paths = FilePaths()
    if args.prepare:
        run_prepare(args, file_paths)
    if args.train:
        train(args, file_paths)
    if args.test:
        test(args, file_paths)
