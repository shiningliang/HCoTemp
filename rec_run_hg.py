import ujson as json
import pickle as pkl
import numpy as np
import logging
import os
import argparse
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import rec_model
from graph_data import parse, build_metapath, build_features
from rec_preprocess import run_prepare
from rec_util import new_train_epoch_hg, new_valid_epoch, load_pkl, AMDataset, my_fn
from pytorch_transformers import WarmupCosineSchedule

from apex import amp
from apex.parallel import DistributedDataParallel

# try:
#     from apex import amp
#
#     _has_apex = True
# except ImportError:
#     _has_apex = False


# def is_apex_available():
#     return _has_apex

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


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
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify gpu device')
    parser.add_argument('--is_distributed', type=bool, default=False,
                        help='distributed training')
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
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')
    train_settings.add_argument('--optim', default='AdamW',
                                help='optimizer type')
    train_settings.add_argument('--warmup', type=float, default=0.5)
    train_settings.add_argument('--patience', type=int, default=2,
                                help='num of epochs for train patients')
    train_settings.add_argument('--loss_batch', type=int, default=2000,
                                help='period to save batch loss')
    train_settings.add_argument('--num_threads', type=int, default=8,
                                help='Number of threads in input pipeline')
    train_settings.add_argument('--local_rank', type=int, default=-1,
                                help='train batch size')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--P', type=int, default=4,
                                help='length of feature period')
    model_settings.add_argument('--T', type=int, default=36,
                                help='length of the year sequence')
    model_settings.add_argument('--NU', type=int, default=26889,
                                help='num of users')
    model_settings.add_argument('--NI', type=int, default=14020,
                                help='num of items')
    # TODO NF是BiLSTM中用户和商品特征embedding维度，若显存不足可尝试减小
    model_settings.add_argument('--NF', type=int, default=128,
                                help='num of factors')
    # TODO NF是BiLSTM中gru维度，若显存不足可尝试减小
    model_settings.add_argument('--n_hidden', type=int, default=128,
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
    model_settings.add_argument('--dynamic', action='store_true',
                                help='if use dynamic embedding')
    model_settings.add_argument('--period', action='store_true',
                                help='if use period embedding')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--task', default='AM_Games',
                               help='the task name')
    path_settings.add_argument('--model', default='Static_ID',
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


def func_train(args, file_paths, gpu, ngpus_per_node):
    torch.cuda.set_device(gpu)
    logger = logging.getLogger('Rec')
    if args.local_rank in [-1, 0]:
        logger.info('Loading record file...')
    # 用户和商品数据
    user_record_file = load_pkl(file_paths.user_record_file)
    item_record_file = load_pkl(file_paths.item_record_file)
    user_length_file = load_pkl(file_paths.user_length_file)
    item_length_file = load_pkl(file_paths.item_length_file)

    train_set = AMDataset(file_paths.train_file, user_record_file, item_record_file, user_length_file, item_length_file,
                          logger, 'train')
    valid_set = AMDataset(file_paths.valid_file, user_record_file, item_record_file, user_length_file, item_length_file,
                          logger, 'valid')

    args.batch_train = int(args.batch_train / ngpus_per_node)
    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_train, shuffle=(train_sampler is None), num_workers=4,
                              collate_fn=my_fn, sampler=train_sampler)
    # train_loader = DataLoader(train_set, batch_size=args.batch_train, num_workers=1,
    #                           collate_fn=my_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_train, shuffle=False, num_workers=4, collate_fn=my_fn)
    train_num = len(train_set.labels)
    valid_num = len(valid_set.labels)

    user_num = len(user_record_file)
    args.NU = user_num
    item_num = len(item_record_file)
    args.NI = item_num
    if args.local_rank in [-1, 0]:
        logger.info('Num of train data {} valid data {}'.format(train_num, valid_num))
        logger.info('Num of users {} items {}'.format(user_num, item_num))

    logger.info('Initialize the model...')
    if args.dynamic:
        UEM = np.random.normal(0., 0.01, (args.T * args.NU + 1, args.NF))
        IEM = np.random.normal(0., 0.01, (args.T * args.NI + 1, args.NF))
    elif args.period:
        UEM = np.random.normal(0., 0.01, (args.P * args.NU + 1, args.NF))
        IEM = np.random.normal(0., 0.01, (args.P * args.NI + 1, args.NF))
    else:
        UEM = np.random.normal(0., 0.01, (args.NU + 1, args.NF))
        IEM = np.random.normal(0., 0.01, (args.NI + 1, args.NF))

    # df = parse(f'./am_data2/Pet/new_metadata2.json')
    logger.info("Build Graph")
    dataset_name = "_".join(args.task.split("_")[:2])
    df = parse("./data/raw_data/{}/new_metadata2.json".format(dataset_name))
    # print(df)
    hg = build_metapath(df).to(args.device)
    features = build_features(df, dataset_name).to(args.device)

    UEM[0] = 0.
    IEM[0] = 0.
    dropout = {'emb': args.emb_dropout, 'layer': args.layer_dropout}
    model = getattr(rec_model, args.model)(features.shape[1], UEM, IEM, args.state, args.T, args.P, args.NU, args.NI,
                                           args.NF,
                                           args.n_class, args.n_hidden,
                                           args.n_layer, dropout, logger).to(args.device)
    # if args.is_distributed:
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[args.local_rank], output_device=args.local_rank,
    # this should be removed if we update BatchNorm stats
    # broadcast_buffers=False)
    # model = torch.nn.DataParallel(model)
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # TODO 显存不足这里尝试开amp, 用O2
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    # model = DistributedDataParallel(model)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=args.patience, verbose=True)
    scheduler = WarmupCosineSchedule(optimizer, args.warmup, (train_num // args.batch_train + 1) * args.epochs)

    # max_acc, max_p, max_r, max_f, max_sum, max_epoch = 0, 0, 0, 0, 0, 0
    # FALSE = {}
    # ROC = {}
    # PRC = {}
    min_loss, min_epoch = 1e10, 0
    for ep in range(1, args.epochs + 1):
        if args.local_rank in [-1, 0]:
            logger.info('Training the model for epoch {}'.format(ep))
        # train_loss = train_one_epoch(model, optimizer, train_num, train_file, user_record_file, item_record_file,
        #                              args, logger)
        train_loss = new_train_epoch_hg(model, optimizer, train_loader, hg, features, args, logger)
        if args.local_rank in [-1, 0]:
            logger.info('Epoch {} MSE {}'.format(ep, train_loss))
        scheduler.step()

        if args.local_rank in [-1, 0]:
            logger.info('Evaluating the model for epoch {}'.format(ep))
        # eval_metrics, fpr, tpr, precision, recall = valid_batch(model, valid_num, args.batch_eval, valid_file,
        #                                                         user_record_file, item_record_file, args.device,
        #                                                         'valid', logger)
        # valid_loss = valid_batch(model, valid_num, args.batch_eval, valid_file, user_record_file, item_record_file,
        #                          args.device)
        valid_loss = new_valid_epoch(model, valid_loader, args)
        if args.local_rank in [-1, 0]:
            logger.info('Valid MSE - {}'.format(valid_loss))
        # logger.info('Valid Loss - {}'.format(eval_metrics['loss']))
        # logger.info('Valid Acc - {}'.format(eval_metrics['acc']))
        # logger.info('Valid Precision - {}'.format(eval_metrics['precision']))
        # logger.info('Valid Recall - {}'.format(eval_metrics['recall']))
        # logger.info('Valid F1 - {}'.format(eval_metrics['f1']))
        # logger.info('Valid AUCROC - {}'.format(eval_metrics['auc_roc']))
        # logger.info('Valid AUCPRC - {}'.format(eval_metrics['auc_prc']))
        # max_acc = max((eval_metrics['acc'], max_acc))
        # max_p = max(eval_metrics['precision'], max_p)
        # max_r = max(eval_metrics['recall'], max_r)
        # max_f = max(eval_metrics['f1'], max_f)
        # valid_sum = eval_metrics['auc_roc'] + eval_metrics['auc_prc']
        # if valid_sum > max_sum:
        #     max_sum = valid_sum
        #     max_epoch = ep
        #     FALSE = {'FP': eval_metrics['fp'], 'FN': eval_metrics['fn']}
        #     ROC = {'FPR': fpr, 'TPR': tpr}
        #     PRC = {'PRECISION': precision, 'RECALL': recall}
        if valid_loss < min_loss:
            min_loss = valid_loss
            min_epoch = ep
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.bin'))

        # scheduler.step(metrics=eval_metrics['f1'])
        # scheduler.step(valid_loss)

    # logger.info('Max Acc - {}'.format(max_acc))
    # logger.info('Max Precision - {}'.format(max_p))
    # logger.info('Max Recall - {}'.format(max_r))
    # logger.info('Max F1 - {}'.format(max_f))
    # logger.info('Max Epoch - {}'.format(max_epoch))
    # logger.info('Max Sum - {}'.format(max_sum))
    if args.local_rank in [-1, 0]:
        logger.info('Min MSE - {}'.format(min_loss))
        logger.info('Min Epoch - {}'.format(min_epoch))
    # with open(os.path.join(args.result_dir, 'FALSE_valid.json'), 'w') as f:
    #     f.write(json.dumps(FALSE) + '\n')
    # f.close()
    # with open(os.path.join(args.result_dir, 'ROC_valid.json'), 'w') as f:
    #     f.write(json.dumps(ROC) + '\n')
    # f.close()
    # with open(os.path.join(args.result_dir, 'PRC_valid.json'), 'w') as f:
    #     f.write(json.dumps(PRC) + '\n')
    # f.close()


def func_test(args, file_paths, gpu, ngpus_per_node):
    torch.cuda.set_device(gpu)
    logger = logging.getLogger('Rec')
    logger.info('Loading record file...')
    user_record_file = load_pkl(file_paths.user_record_file)
    item_record_file = load_pkl(file_paths.item_record_file)
    user_length_file = load_pkl(file_paths.user_length_file)
    item_length_file = load_pkl(file_paths.item_length_file)

    test_set = AMDataset(file_paths.test_file, user_record_file, item_record_file, user_length_file, item_length_file,
                         logger, 'test')
    args.batch_eval = int(args.batch_eval / ngpus_per_node)
    test_loader = DataLoader(test_set, args.batch_eval, num_workers=4, collate_fn=my_fn)
    test_num = len(test_set.labels)
    logger.info('Num of test data {}'.format(test_num))
    user_num = len(user_record_file)
    args.NU = user_num
    item_num = len(item_record_file)
    args.NI = item_num
    logger.info('Num of users {} items {}'.format(user_num, item_num))

    logger.info('Initialize the model...')
    if args.dynamic:
        UEM = np.random.normal(0., 0.01, (args.T * args.NU + 1, args.NF))
        IEM = np.random.normal(0., 0.01, (args.T * args.NI + 1, args.NF))
    elif args.period:
        UEM = np.random.normal(0., 0.01, (args.P * args.NU + 1, args.NF))
        IEM = np.random.normal(0., 0.01, (args.P * args.NI + 1, args.NF))
    else:
        UEM = np.random.normal(0., 0.01, (args.NU + 1, args.NF))
        IEM = np.random.normal(0., 0.01, (args.NI + 1, args.NF))

    UEM[0] = 0.
    IEM[0] = 0.
    dropout = {'emb': args.emb_dropout, 'layer': args.layer_dropout}
    model = getattr(rec_model, args.model)(UEM, IEM, args.state, args.T, args.P, args.NU, args.NI, args.NF,
                                           args.n_class, args.n_hidden,
                                           args.n_layer, dropout, logger).to(args.device)
    # if args.is_distributed:
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[args.local_rank], output_device=args.local_rank,
    # this should be removed if we update BatchNorm stats
    # broadcast_buffers=False)
    # model = torch.nn.DataParallel(model)
    # optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    # model = DistributedDataParallel(model)
    logger.info(args.model_dir)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model.bin')))

    # eval_metrics, fpr, tpr, precision, recall = valid_batch(model, test_num, args.batch_eval, test_file,
    #                                                         user_record_file, item_record_file, args.device,
    #                                                         'test', logger)
    # test_loss = valid_batch(model, test_num, args.batch_eval, test_file, user_record_file, item_record_file,
    #                         args.device)
    test_loss = new_valid_epoch(model, test_loader, args)
    logger.info('Test MSE - {}'.format(test_loss))
    # logger.info('Test Acc - {}'.format(eval_metrics['acc']))
    # logger.info('Test Precision - {}'.format(eval_metrics['precision']))
    # logger.info('Test Recall - {}'.format(eval_metrics['recall']))
    # logger.info('Test F1 - {}'.format(eval_metrics['f1']))
    # logger.info('Test AUCROC - {}'.format(eval_metrics['auc_roc']))
    # logger.info('Test AUCPRC - {}'.format(eval_metrics['auc_prc']))

    # FALSE = {'FP': eval_metrics['fp'], 'FN': eval_metrics['fn']}
    # ROC = {'FPR': fpr, 'TPR': tpr}
    # PRC = {'PRECISION': precision, 'RECALL': recall}
    #
    # with open(os.path.join(args.result_dir, 'FALSE_test.json'), 'w') as f:
    #     f.write(json.dumps(FALSE) + '\n')
    # f.close()
    # with open(os.path.join(args.result_dir, 'ROC_test.json'), 'w') as f:
    #     f.write(json.dumps(ROC) + '\n')
    # f.close()
    # with open(os.path.join(args.result_dir, 'PRC_test.json'), 'w') as f:
    #     f.write(json.dumps(PRC) + '\n')
    # f.close()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


if __name__ == '__main__':
    args = parse_args()

    logger = logging.getLogger('Rec')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # WORLD_SIZE 由torch.distributed.launch.py产生 具体数值为 nproc_per_node*node(主机数，这里为1)
    # num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # args.is_distributed = num_gpus > 1
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.gpu > -1:
        args.device = torch.device('cuda')
        # torch.cuda.set_device(args.gpu)
        # torch.distributed.init_process_group(backend="nccl", init_method="env://")
        # synchronize()
    else:
        args.device = torch.device('cpu')

    logger.info('Preparing the directories...')
    args.raw_dir = os.path.join(args.raw_dir, args.task)
    if args.dynamic:
        args.task = args.task + '_dynamic'
    elif args.period:
        args.task = args.task + '_period'
    else:
        args.task = args.task + '_static'
    args.processed_dir = os.path.join(args.processed_dir, args.task)
    args.model_dir = os.path.join(args.outputs_dir, args.task, args.model, args.model_dir)
    args.result_dir = os.path.join(args.outputs_dir, args.task, args.model, args.result_dir)
    args.pics_dir = os.path.join(args.outputs_dir, args.task, args.model, args.pics_dir)
    args.summary_dir = os.path.join(args.outputs_dir, args.task, args.model, args.summary_dir)

    # 构建输出文件夹
    for dir_path in [args.raw_dir, args.processed_dir, args.model_dir, args.result_dir, args.pics_dir,
     args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 预处理后的数据
    class FilePaths(object):
        def __init__(self):
            self.train_file = os.path.join(args.processed_dir, 'train.pkl')
            self.valid_file = os.path.join(args.processed_dir, 'valid.pkl')
            self.test_file = os.path.join(args.processed_dir, 'test.pkl')
            self.user_record_file = os.path.join(args.processed_dir, 'user_record.pkl')
            self.item_record_file = os.path.join(args.processed_dir, 'item_record.pkl')
            self.user_length_file = os.path.join(args.processed_dir, 'user_length.pkl')
            self.item_length_file = os.path.join(args.processed_dir, 'item_length.pkl')


    args.state = "static"
    if args.dynamic:
        args.state = "dynamic"
    elif args.period:
        args.state = "period"
    logger.info('Running with args : {}'.format(args))
    file_paths = FilePaths()
    if args.prepare:
        run_prepare(args, file_paths)
    # 训练入口
    if args.train:
        func_train(args, file_paths, args.local_rank, 2)
    # 测试入口
    if args.test:
        func_test(args, file_paths, args.local_rank, 2)
