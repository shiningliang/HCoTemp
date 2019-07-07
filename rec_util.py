import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, \
    precision_recall_curve
import matplotlib.pyplot as plt
import seaborn
import os

seaborn.set_context(context="talk")
plt.switch_backend('agg')


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
        if len(urec) > 20:
            print(uid)
        u_records.append(urec)
        tmp_len = max((len(r) for r in urec))
        u_maxlen = tmp_len if tmp_len > u_maxlen else u_maxlen

        irec = item_records[iid - 1][str(iid)]
        if len(irec) > 20:
            print(iid)
        i_records.append(irec)
        tmp_len = max((len(r) for r in irec))
        i_maxlen = tmp_len if tmp_len > i_maxlen else i_maxlen

    u_records = list(list(map(lambda l: l + [0] * (u_maxlen - len(l)), records)) for records in u_records)
    i_records = list(list(map(lambda l: l + [0] * (i_maxlen - len(l)), records)) for records in i_records)
    u_records = np.asarray(u_records, dtype=np.int64)

    i_records = np.asarray(i_records, dtype=np.int64)
    labels = np.asarray(labels, np.int64)

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
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, b_labels)
        loss.backward()
        if args.clip > 0:
            # 梯度裁剪，输入是(NN参数，最大梯度范数，范数类型=2)，一般默认为L2范数
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        n_batch_loss += loss.item()
        bidx = batch_idx + 1
        if bidx % args.period == 0:
            logger.info('AvgLoss batch [{} {}] - {}'.format(bidx - args.period + 1, bidx, n_batch_loss / args.period))
            n_batch_loss = 0
        train_loss.append(loss.item())

    avg_train_loss = np.mean(train_loss)
    return avg_train_loss


def valid_batch(model, data_num, batch_size, valid_file, user_records, item_records, device, data_type, logger):
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

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(rec_outputs, b_labels)
        losses.append(loss.item())
        rec_preds = torch.max(rec_outputs.cpu(), 1)[1].numpy()
        rec_scores = rec_outputs.cpu()[:, 1].numpy()
        b_labels = b_labels.cpu().numpy()
        preds += rec_preds.tolist()
        scores += rec_scores.tolist()
        labels += b_labels.tolist()
        # if data_type == 'valid' or data_type == 'test':
        #     for pred, label, eid in zip(rec_preds, b_labels, eids):
        #         if label == 1 and pred == 0:
        #             fn.append(eid)
        #         if label == 0 and pred == 1:
        #             fp.append(eid)

    metrics['loss'] = np.mean(losses)
    metrics['acc'] = accuracy_score(labels, preds)
    metrics['precision'] = precision_score(labels, preds)
    metrics['recall'] = recall_score(labels, preds)
    metrics['f1'] = f1_score(labels, preds)
    fpr, tpr, _ = roc_curve(labels, scores)
    (precisions, recalls, _) = precision_recall_curve(labels, scores)
    metrics['auc_roc'] = auc(fpr, tpr)
    metrics['auc_prc'] = auc(recalls, precisions)
    if data_type == 'valid' or data_type == 'test':
        metrics['fp'] = fp
        metrics['fn'] = fn
    logger.info('Full confusion matrix')
    logger.info(confusion_matrix(labels, preds))
    return metrics, fpr, tpr, precisions, recalls
