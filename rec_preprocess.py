import ujson as json
import pickle as pkl
import os
import pandas as pd
from rec_util import dump_pkl


class BasicParser(object):
    def __init__(self, ):


def parse_dynamic_record(file_path, user_in, item_in, T, user_out, item_out, data_type):
    print("Generating {} samples...".format(data_type))
    user_path = os.path.join(file_path, user_in)
    ulines = open(user_path, 'r').readlines()
    item_path = os.path.join(file_path, item_in)
    ilines = open(item_path, 'r').readlines()
    NU = len(ulines)
    NI = len(ilines)
    print(NU, NI)

    urecords = []
    for line in ulines:
        urecords.append(json.loads(line))
    for uid, record in enumerate(urecords):
        uid = str(uid + 1)
        u_len = len(record[uid])
        if u_len >= T:
            record[uid] = record[uid][:T]
            u_len = T
        else:
            for i in range(T - u_len):
                record[uid].append([0])
        for t in range(u_len):
            # record[uid][t] 第t月与user有交互的item list
            for idx, ut_i in enumerate(record[uid][t]):
                record[uid][t][idx] = t * NI + ut_i  # 按第t月寻找组 按id偏移

    irecords = []
    for line in ilines:
        irecords.append(json.loads(line))
    for iid, record in enumerate(irecords):
        iid = str(iid + 1)
        i_len = len(record[iid])
        if i_len >= T:
            record[iid] = record[iid][:T]
            i_len = T
        else:
            for i in range(T - i_len):
                record[iid].append([0])
        for t in range(i_len):
            for idx, it_u in enumerate(record[iid][t]):
                record[iid][t][idx] = t * NU + it_u

    dump_pkl(user_out, urecords)
    dump_pkl(item_out, irecords)

    return NU, NI


def parse_static_record(file_path, user_in, item_in, T, user_out, item_out, ulen_out, ilen_out, data_type):
    print("Generating {} samples...".format(data_type))
    user_path = os.path.join(file_path, user_in)
    ulines = open(user_path, 'r').readlines()
    item_path = os.path.join(file_path, item_in)
    ilines = open(item_path, 'r').readlines()
    NU = len(ulines)
    NI = len(ilines)
    print(NU, NI)

    urecords, urec_length = [], []
    for line in ulines:
        urecords.append(json.loads(line))
    for uid, record in enumerate(urecords):
        uid = str(uid + 1)
        u_len = len(record[uid])
        if u_len >= T:
            record[uid] = record[uid][:T]
            u_len = T
        else:
            for i in range(T - u_len):
                record[uid].append([0])
        urec_length.append(u_len)

    irecords, irec_length = [], []
    for line in ilines:
        irecords.append(json.loads(line))
    for iid, record in enumerate(irecords):
        iid = str(iid + 1)
        i_len = len(record[iid])
        if i_len >= T:
            record[iid] = record[iid][:T]
            i_len = T
        else:
            for i in range(T - i_len):
                record[iid].append([0])
        irec_length.append(i_len)

    dump_pkl(user_out, urecords)
    dump_pkl(item_out, irecords)
    dump_pkl(ulen_out, urec_length)
    dump_pkl(ilen_out, irec_length)


def parse_period_record(file_path, user_in, item_in, T, period, user_out, item_out, data_type):
    print("Generating {} samples...".format(data_type))
    user_path = os.path.join(file_path, user_in)
    ulines = open(user_path, 'r').readlines()
    item_path = os.path.join(file_path, item_in)
    ilines = open(item_path, 'r').readlines()
    NU = len(ulines)
    NI = len(ilines)
    print(NU, NI)

    urecords = []
    for line in ulines:
        urecords.append(json.loads(line))
    for uid, record in enumerate(urecords):
        uid = str(uid + 1)
        u_len = len(record[uid])
        if u_len >= T:
            record[uid] = record[uid][:T]
            u_len = T
        else:
            for i in range(T - u_len):
                record[uid].append([0])
        for t in range(u_len):
            # record[uid][t] 第t月与user有交互的item list
            for idx, ut_i in enumerate(record[uid][t]):
                record[uid][t][idx] = t % period * NI + ut_i  # 按第t月寻找组 按id偏移

    irecords = []
    for line in ilines:
        irecords.append(json.loads(line))
    for iid, record in enumerate(irecords):
        iid = str(iid + 1)
        i_len = len(record[iid])
        if i_len >= T:
            record[iid] = record[iid][:T]
            i_len = T
        else:
            for i in range(T - i_len):
                record[iid].append([0])
        for t in range(i_len):
            for idx, it_u in enumerate(record[iid][t]):
                record[iid][t][idx] = t % period * NU + it_u

    dump_pkl(user_out, urecords)
    dump_pkl(item_out, irecords)

    return NU, NI


def parse_dynamic_set(file_path, name_in, T, NU, NI, name_out, data_type):
    print("Generating {} samples...".format(data_type))

    full_path = os.path.join(file_path, name_in)
    raw = pd.read_csv(full_path, sep=',')
    uids, iids, labels = [], [], []
    for i, row in raw.iterrows():
        uid, iid, label = row['uids'], row['iids'], row['labels']
        uids.append([t * NU + uid for t in range(T)])
        iids.append([t * NI + iid for t in range(T)])
        labels.append(label)

    processed = {'uids': uids, 'iids': iids, 'labels': labels}
    dump_pkl(name_out, processed)


def parse_set(file_path, name_in, T, name_out, data_type):
    print("Generating {} samples...".format(data_type))

    full_path = os.path.join(file_path, name_in)
    raw = pd.read_csv(full_path, sep=',')
    uids, iids, labels = [], [], []
    for i, row in raw.iterrows():
        uid, iid, label = row['uids'], row['iids'], row['labels']
        uids.append([uid] * T)
        iids.append([iid] * T)
        labels.append(label)

    processed = {'uids': uids, 'iids': iids, 'labels': labels}
    dump_pkl(name_out, processed)


def parse_period_set(file_path, name_in, T, period, NU, NI, name_out, data_type):
    print("Generating {} samples...".format(data_type))

    full_path = os.path.join(file_path, name_in)
    raw = pd.read_csv(full_path, sep=',')
    uids, iids, labels = [], [], []
    for i, row in raw.iterrows():
        uid, iid, label = row['uids'], row['iids'], row['labels']
        uids.append([t % period * NU + uid for t in range(T)])
        iids.append([t % period * NI + iid for t in range(T)])
        labels.append(label)

    processed = {'uids': uids, 'iids': iids, 'labels': labels}
    dump_pkl(name_out, processed)


def run_prepare(config, flags):
    if config.dynamic:
        num_user, num_item = parse_dynamic_record(config.raw_dir, config.user_record_file, config.item_record_file,
                                                  config.T, flags.user_record_file, flags.item_record_file, 'record')
        parse_dynamic_set(config.raw_dir, config.train_file, config.T, num_user, num_item, flags.train_file, 'train')
        parse_dynamic_set(config.raw_dir, config.valid_file, config.T, num_user, num_item, flags.valid_file, 'valid')
        parse_dynamic_set(config.raw_dir, config.test_file, config.T, num_user, num_item, flags.test_file, 'test')
    elif config.period:
        num_user, num_item = parse_period_record(config.raw_dir, config.user_record_file, config.item_record_file,
                                                 config.T, config.P, flags.user_record_file,
                                                 flags.item_record_file, 'record')
        parse_period_set(config.raw_dir, config.train_file, config.T, config.P, num_user, num_item, flags.train_file,
                         'train')
        parse_period_set(config.raw_dir, config.valid_file, config.T, config.P, num_user, num_item, flags.valid_file,
                         'valid')
        parse_period_set(config.raw_dir, config.test_file, config.T, config.P, num_user, num_item, flags.test_file,
                         'test')
    else:
        parse_static_record(config.raw_dir, config.user_record_file, config.item_record_file, config.T,
                            flags.user_record_file, flags.item_record_file,
                            flags.user_length_file, flags.item_length_file, 'record')
        parse_set(config.raw_dir, config.train_file, config.T, flags.train_file, 'train')
        parse_set(config.raw_dir, config.valid_file, config.T, flags.valid_file, 'valid')
        parse_set(config.raw_dir, config.test_file, config.T, flags.test_file, 'test')
