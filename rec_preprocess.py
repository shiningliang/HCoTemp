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

    for record in urecords:
        user = record['user']
        for t in range(T):
            ut = user[t]  # 第t月与user有交互的item list
            for idx, ut_i in enumerate(ut):
                ut[idx] = t * NI + ut_i  # 按第t月寻找组 按id偏移

    item_path = os.path.join(file_path, item_in)
    lines = open(item_path, 'r').readlines()
    irecords = []
    for line in lines:
        irecords.append(json.loads(line))
    for record in irecords:
        item = record['item']
        for t in range(T):
            it = item[t]
            for idx, it_u in enumerate(it):
                it[idx] = t * NU + it_u

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
