import os
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, \
    precision_recall_curve

# import matplotlib.pyplot as plt
# import seaborn

try:
    from apex import amp
except:
    print("Please install apex")


# seaborn.set_context(context="talk")
# plt.switch_backend('agg')


def get_batch(uids, iids, labels, user_records, item_records, device):
    uids = np.asarray(uids, dtype=np.int64)
    iids = np.asarray(iids, dtype=np.int64)

    raw_uids = uids[:, 0]
    raw_iids = iids[:, 0]

    u_records, i_records = [], []
    u_maxlen = 0
    i_maxlen = 0
    for uid, iid in zip(raw_uids, raw_iids):
        urec = user_records[uid - 1][str(uid)]
        u_records.append(urec)
        tmp_len = max((len(r) for r in urec))
        u_maxlen = tmp_len if tmp_len > u_maxlen else u_maxlen

        irec = item_records[iid - 1][str(iid)]
        i_records.append(irec)
        tmp_len = max((len(r) for r in irec))
        i_maxlen = tmp_len if tmp_len > i_maxlen else i_maxlen

    u_records = list(list(map(lambda l: l + [0] * (u_maxlen - len(l)), records)) for records in u_records)
    i_records = list(list(map(lambda l: l + [0] * (i_maxlen - len(l)), records)) for records in i_records)
    u_records = np.asarray(u_records, dtype=np.int64)

    i_records = np.asarray(i_records, dtype=np.int64)
    labels = np.asarray(labels, np.float32)

    return torch.from_numpy(u_records).to(device), torch.from_numpy(i_records).to(device), \
           torch.from_numpy(uids).to(device), torch.from_numpy(iids).to(device), torch.from_numpy(labels).to(device)


def train_one_epoch(model, optimizer, train_num, train_file, user_records, item_records, args, logger):
    model.train()
    train_loss = []
    n_batch_loss = 0
    for batch_idx, batch in enumerate(range(0, train_num, args.batch_train)):
        start_idx = batch
        end_idx = start_idx + args.batch_train
        b_user_records, b_item_records, b_uids, b_iids, b_labels = get_batch(train_file['uids'][start_idx:end_idx],
                                                                             train_file['iids'][start_idx:end_idx],
                                                                             train_file['labels'][start_idx:end_idx],
                                                                             user_records, item_records,
                                                                             args.device)

        optimizer.zero_grad()
        outputs = model(b_user_records, b_item_records, b_uids, b_iids)
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, b_labels.reshape(b_labels.shape[0], 1))
        loss.backward()
        if args.clip > 0:
            # 梯度裁剪，输入是(NN参数，最大梯度范数，范数类型=2)，一般默认为L2范数
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        n_batch_loss += loss.item()
        bidx = batch_idx + 1
        if bidx % args.loss_batch == 0:
            logger.info(
                'AvgLoss batch [{} {}] - {}'.format(bidx - args.loss_batch + 1, bidx, n_batch_loss / args.loss_batch))
            n_batch_loss = 0
        train_loss.append(loss.item())

    avg_train_loss = np.mean(train_loss)
    return avg_train_loss


def valid_batch(model, data_num, batch_size, valid_file, user_records, item_records, device):
    losses = []
    fp, fn = [], []
    preds, scores, labels = [], [], []
    metrics = {}
    model.eval()
    for batch_idx, batch in enumerate(range(0, data_num, batch_size)):
        start_idx = batch
        end_idx = start_idx + batch_size
        b_user_records, b_item_records, b_uids, b_iids, b_labels = get_batch(valid_file['uids'][start_idx:end_idx],
                                                                             valid_file['iids'][start_idx:end_idx],
                                                                             valid_file['labels'][start_idx:end_idx],
                                                                             user_records, item_records,
                                                                             device)
        rec_outputs = model(b_user_records, b_item_records, b_uids, b_iids)
        rec_outputs = rec_outputs.detach()

        # criterion = torch.nn.CrossEntropyLoss()
        criterion = torch.nn.MSELoss()
        loss = criterion(rec_outputs, b_labels.reshape(b_labels.shape[0], 1))
        losses.append(loss.item())
        # rec_preds = torch.max(rec_outputs.cpu(), 1)[1].numpy()
        # rec_scores = rec_outputs.cpu()[:, 1].numpy()
        # b_labels = b_labels.cpu().numpy()
        # preds += rec_preds.tolist()
        # scores += rec_scores.tolist()
        # labels += b_labels.tolist()
        # if data_type == 'valid' or data_type == 'test':
        #     for pred, label, eid in zip(rec_preds, b_labels, eids):
        #         if label == 1 and pred == 0:
        #             fn.append(eid)
        #         if label == 0 and pred == 1:
        #             fp.append(eid)

    return np.mean(losses)
    # metrics['loss'] = np.mean(losses)
    # metrics['acc'] = accuracy_score(labels, preds)
    # metrics['precision'] = precision_score(labels, preds)
    # metrics['recall'] = recall_score(labels, preds)
    # metrics['f1'] = f1_score(labels, preds)
    # fpr, tpr, _ = roc_curve(labels, scores)
    # (precisions, recalls, _) = precision_recall_curve(labels, scores)
    # metrics['auc_roc'] = auc(fpr, tpr)
    # metrics['auc_prc'] = auc(recalls, precisions)
    # if data_type == 'valid' or data_type == 'test':
    #     metrics['fp'] = fp
    #     metrics['fn'] = fn
    # logger.info('Full confusion matrix')
    # logger.info(confusion_matrix(labels, preds))
    # return metrics, fpr, tpr, precisions, recalls


def load_pkl(path):
    with open(path, 'rb') as f:
        data = joblib.load(f)
    f.close()

    return data


def dump_pkl(path, obj):
    with open(path, 'wb') as f:
        joblib.dump(obj, f)
    f.close()


class AMDataset(Dataset):
    def __init__(self, data_path, user_rec, item_rec, user_length, item_length, logger, data_type):
        logger.info('Loading {} file'.format(data_type))
        raw_data = load_pkl(data_path)
        self.uids = np.asarray(raw_data['uids'], dtype=np.int64)
        self.iids = np.asarray(raw_data['iids'], dtype=np.int64)
        self.labels = np.asarray(raw_data['labels'], np.float32)
        self.user_records = user_rec
        self.item_records = item_rec
        self.user_lengths = user_length
        self.item_lengths = item_length
        del raw_data

    def __getitem__(self, index):
        uid = self.uids[index][0]
        urec = self.user_records[uid - 1][str(uid)]
        iid = self.iids[index][0]
        irec = self.item_records[iid - 1][str(iid)]

        return self.uids[index], self.iids[index], urec, irec, self.user_lengths[uid - 1], self.item_lengths[iid - 1], \
               self.labels[index]

    def __len__(self):
        return len(self.labels)


def my_fn(batch):
    uids, iids, u_records, i_records, u_lengths, i_lengths, labels = zip(*batch)
    u_maxlen, i_maxlen = 0, 0
    for urec, irec in zip(u_records, i_records):
        tmp_len = max((len(r) for r in urec))
        u_maxlen = tmp_len if tmp_len > u_maxlen else u_maxlen
        tmp_len = max((len(r) for r in irec))
        i_maxlen = tmp_len if tmp_len > i_maxlen else i_maxlen

    u_records = list(list(map(lambda l: l + [0] * (u_maxlen - len(l)), records)) for records in u_records)
    i_records = list(list(map(lambda l: l + [0] * (i_maxlen - len(l)), records)) for records in i_records)
    u_records = np.asarray(u_records, dtype=np.int64)
    i_records = np.asarray(i_records, dtype=np.int64)

    return torch.from_numpy(u_records), torch.from_numpy(i_records), torch.from_numpy(np.asarray(uids)), \
           torch.from_numpy(np.asarray(iids)), torch.from_numpy(np.asarray(u_lengths)), \
           torch.from_numpy(np.asarray(i_lengths)), torch.from_numpy(np.asarray(labels))


def new_train_epoch(model, optimizer, loader, args, logger, scheduler=None, is_dist=True):
    model.train()
    train_loss = []
    n_batch_loss = 0
    for batch_idx, batch in enumerate(loader):
        b_user_records, b_item_records, b_uids, b_iids, b_ulens, b_ilens, b_labels = batch
        b_user_records = b_user_records.to(args.device)
        b_item_records = b_item_records.to(args.device)
        b_uids = b_uids.to(args.device)
        b_iids = b_iids.to(args.device)
        b_ulens = b_ulens.to(args.device)
        b_ilens = b_ilens.to(args.device)
        b_labels = b_labels.to(args.device)
        optimizer.zero_grad()
        outputs = model(b_user_records, b_item_records, b_uids, b_iids, b_ulens, b_ilens)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, b_labels.reshape(b_labels.shape[0], 1))
        if is_dist:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if args.clip > -1:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
        optimizer.step()
        if scheduler:
            scheduler.step()

        n_batch_loss += loss.item()
        bidx = batch_idx + 1
        # logger.info('AvgLoss batch [{}] - {}'.format(bidx, n_batch_loss / bidx))
        if bidx % args.loss_batch == 0:
            if args.local_rank in [-1, 0]:
                logger.info(
                    'AvgLoss batch [{} {}] - {}'.format(bidx - args.loss_batch + 1, bidx, n_batch_loss / args.loss_batch))
            n_batch_loss = 0

        train_loss.append(loss.item())

    return np.mean(train_loss)


def new_train_epoch_hg(model, optimizer, loader, hg, features, args, logger, is_dist=True):
    model.train()
    train_loss = []
    n_batch_loss = 0
    for batch_idx, batch in enumerate(loader):
        b_user_records, b_item_records, b_uids, b_iids, b_ulens, b_ilens, b_labels = batch
        b_user_records = b_user_records.to(args.device)
        b_item_records = b_item_records.to(args.device)
        b_uids = b_uids.to(args.device)
        b_iids = b_iids.to(args.device)
        b_ulens = b_ulens.to(args.device)
        b_ilens = b_ilens.to(args.device)
        b_labels = b_labels.to(args.device)
        optimizer.zero_grad()
        outputs = model(b_user_records, b_item_records, b_uids, b_iids, b_ulens, b_ilens, hg, features)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, b_labels.reshape(b_labels.shape[0], 1))
        # if is_dist:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss.backward()
        # if args.clip > -1:
        #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
        loss.backward()
        optimizer.step()
        n_batch_loss += loss.item()
        bidx = batch_idx + 1
        logger.info(
            'AvgLoss batch [{}] - {}'.format(bidx, n_batch_loss / bidx))
        if bidx % args.loss_batch == 0:
            if args.local_rank in [-1, 0]:
                logger.info(
                    'AvgLoss batch [{} {}] - {}'.format(bidx - args.loss_batch + 1, bidx,
                                                        n_batch_loss / args.loss_batch))
            n_batch_loss = 0

        train_loss.append(loss.item())

    return np.mean(train_loss)


def new_valid_epoch(model, loader, args):
    valid_loss = []
    model.eval()
    for batch_idx, batch in enumerate(loader):
        b_user_records, b_item_records, b_uids, b_iids, b_ulens, b_ilens, b_labels = batch
        b_user_records = b_user_records.to(args.device)
        b_item_records = b_item_records.to(args.device)
        b_uids = b_uids.to(args.device)
        b_iids = b_iids.to(args.device)
        b_ulens = b_ulens.to(args.device)
        b_ilens = b_ilens.to(args.device)
        b_labels = b_labels.to(args.device)
        outputs = model(b_user_records, b_item_records, b_uids, b_iids, b_ulens, b_ilens)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, b_labels.reshape(b_labels.shape[0], 1))
        valid_loss.append(loss.item())

    return np.mean(valid_loss)
