import ujson as json
import pickle as pkl
import os
import pandas as pd


def parse_record(file_path, user_in, item_in, T, NU, NI, user_out, item_out, data_type):
    print("Generating {} samples...".format(data_type))
    user_path = os.path.join(file_path, user_in)
    lines = open(user_path, 'r').readlines()
    urecords = []
    for line in lines:
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
        if len(record[uid]) != 20:
            print(uid)

    item_path = os.path.join(file_path, item_in)
    lines = open(item_path, 'r').readlines()
    irecords = []
    for line in lines:
        irecords.append(json.loads(line))
    for iid, record in enumerate(irecords):
        iid = str(iid + 1)
        i_len = len(record[iid])
        if iid == 1846:
            print(iid)
        if i_len >= T:
            record[iid] = record[iid][:T]
            i_len = T
        else:
            for i in range(T - i_len):
                record[iid].append([0])
        for t in range(i_len):
            for idx, it_u in enumerate(record[iid][t]):
                record[iid][t][idx] = t * NU + it_u
        if len(record[iid]) != 20:
            print(iid)

    with open(user_out, 'wb') as fo:
        pkl.dump(urecords, fo)
    fo.close()

    with open(item_out, 'wb') as fo:
        pkl.dump(irecords, fo)
    fo.close()


def parse_set(file_path, name_in, T, NU, NI, name_out, data_type):
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
    with open(name_out, 'wb') as fo:
        pkl.dump(processed, fo)
    fo.close()


def run_prepare(config, flags):
    parse_record(config.raw_dir, config.user_record_file, config.item_record_file, config.T, config.NU, config.NI,
                 flags.user_record_file, flags.item_record_file, 'record')

    parse_set(config.raw_dir, config.train_file, config.T, config.NU, config.NI, flags.train_file, 'train')
    parse_set(config.raw_dir, config.valid_file, config.T, config.NU, config.NI, flags.valid_file, 'valid')
    parse_set(config.raw_dir, config.test_file, config.T, config.NU, config.NI, flags.test_file, 'test')
