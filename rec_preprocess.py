import ujson as json
import pickle as pkl
import os
import pandas as pd
from rec_util import dump_pkl
from clean_data import show_len


class BasicParser(object):
    def __init__(self, file_path, user_in, item_in, T, P, data_type):
        print("Generating {} samples...".format(data_type))
        user_path = os.path.join(file_path, user_in)
        self.ulines = open(user_path, 'r').readlines()
        item_path = os.path.join(file_path, item_in)
        self.ilines = open(item_path, 'r').readlines()
        self.NU = len(self.ulines)
        self.NI = len(self.ilines)
        self.T = T
        self.P = P
        print(self.NU, self.NI)

    def parse_record(self, user_out, item_out, ulen_out, ilen_out):
        urecords, urec_length = [], []
        for line in self.ulines:
            urecords.append(json.loads(line))
        for uid, record in enumerate(urecords):
            uid = str(uid + 1)
            u_len = len(record[uid])
            if u_len == 0:
                print(uid)
            if u_len >= self.T:
                record[uid] = record[uid][:self.T]
                u_len = self.T
            else:
                for i in range(self.T - u_len):
                    record[uid].append([0])
            urec_length.append(u_len)

        irecords, irec_length = [], []
        for line in self.ilines:
            irecords.append(json.loads(line))
        for iid, record in enumerate(irecords):
            iid = str(iid + 1)
            i_len = len(record[iid])
            if i_len == 0:
                print(iid)
            if i_len >= self.T:
                record[iid] = record[iid][:self.T]
                i_len = self.T
            else:
                for i in range(self.T - i_len):
                    record[iid].append([0])
            irec_length.append(i_len)

        dump_pkl(user_out, urecords)
        dump_pkl(item_out, irecords)
        dump_pkl(ulen_out, urec_length)
        dump_pkl(ilen_out, irec_length)

    def parse_set(self, file_path, name_in, name_out, data_type, task):
        print("Generating {} samples...".format(data_type))

        full_path = os.path.join(file_path, name_in)
        raw = pd.read_csv(full_path, sep=',')
        uids, iids, labels = [], [], []
        for i, row in raw.iterrows():
            uid, iid, label = row['uids'], row['iids'], row['labels']
            if task == 'static':
                uids.append([uid] * self.T)
                iids.append([iid] * self.T)
            elif task == 'dynamic':
                uids.append([t * self.NU + uid for t in range(self.T)])
                iids.append([t * self.NI + iid for t in range(self.T)])
            else:
                uids.append([t % self.P * self.NU + uid for t in range(self.T)])
                iids.append([t % self.P * self.NI + iid for t in range(self.T)])
            labels.append(label)

        processed = {'uids': uids, 'iids': iids, 'labels': labels}
        dump_pkl(name_out, processed)


class DynamicParser(BasicParser):
    def parse_record(self, user_out, item_out, ulen_out, ilen_out):
        urecords, urec_length = [], []
        for line in self.ulines:
            urecords.append(json.loads(line))
        for uid, record in enumerate(urecords):
            uid = str(uid + 1)
            u_len = len(record[uid])
            # u_raw_length.append(u_len)
            if u_len >= self.T:
                record[uid] = record[uid][:self.T]
                u_len = self.T
            else:
                for i in range(self.T - u_len):
                    record[uid].append([0])
            for t in range(u_len):
                # record[uid][t] 第t月与user有交互的item list
                for idx, ut_i in enumerate(record[uid][t]):
                    record[uid][t][idx] = t * self.NI + ut_i  # 按第t月寻找组 按id偏移
            urec_length.append(u_len)

        irecords, irec_length = [], []
        for line in self.ilines:
            irecords.append(json.loads(line))
        for iid, record in enumerate(irecords):
            iid = str(iid + 1)
            i_len = len(record[iid])
            # i_raw_length.append(i_len)
            if i_len >= self.T:
                record[iid] = record[iid][: self.T]
                i_len = self.T
            else:
                for i in range(self.T - i_len):
                    record[iid].append([0])
            for t in range(i_len):
                for idx, it_u in enumerate(record[iid][t]):
                    record[iid][t][idx] = t * self.NU + it_u
            irec_length.append(i_len)

        dump_pkl(user_out, urecords)
        dump_pkl(item_out, irecords)
        dump_pkl(ulen_out, urec_length)
        dump_pkl(ilen_out, irec_length)


class PeriodParser(BasicParser):
    def parse_record(self, user_out, item_out, ulen_out, ilen_out):
        urecords, urec_length = [], []
        for line in self.ulines:
            urecords.append(json.loads(line))
        for uid, record in enumerate(urecords):
            uid = str(uid + 1)
            u_len = len(record[uid])
            # u_raw_length.append(u_len)
            if u_len >= self.T:
                record[uid] = record[uid][:self.T]
                u_len = self.T
            else:
                for i in range(self.T - u_len):
                    record[uid].append([0])
            for t in range(u_len):
                for idx, ut_i in enumerate(record[uid][t]):
                    record[uid][t][idx] = t % self.P * self.NI + ut_i  # 按第t月寻找组 按id偏移
            urec_length.append(u_len)

        irecords, irec_length = [], []
        for line in self.ilines:
            irecords.append(json.loads(line))
        for iid, record in enumerate(irecords):
            iid = str(iid + 1)
            i_len = len(record[iid])
            # i_raw_length.append(i_len)
            if i_len >= self.T:
                record[iid] = record[iid][:self.T]
                i_len = self.T
            else:
                for i in range(self.T - i_len):
                    record[iid].append([0])
            for t in range(i_len):
                for idx, it_u in enumerate(record[iid][t]):
                    record[iid][t][idx] = t % self.P * self.NU + it_u
            irec_length.append(i_len)

        dump_pkl(user_out, urecords)
        dump_pkl(item_out, irecords)
        dump_pkl(ulen_out, urec_length)
        dump_pkl(ilen_out, irec_length)


def run_prepare(config, flags):
    if config.dynamic:
        parser = DynamicParser(config.raw_dir, config.user_record_file, config.item_record_file,
                               config.T, config.P, 'record')
        parser.parse_record(flags.user_record_file, flags.item_record_file,
                            flags.user_length_file, flags.item_length_file)
        task = 'dynamic'
        # num_user, num_item = dynamic_parser.NU, dynamic_parser.NI
        # parse_dynamic_set(config.raw_dir, config.train_file, config.T, num_user, num_item, flags.train_file, 'train')
        # parse_dynamic_set(config.raw_dir, config.valid_file, config.T, num_user, num_item, flags.valid_file, 'valid')
        # parse_dynamic_set(config.raw_dir, config.test_file, config.T, num_user, num_item, flags.test_file, 'test')
    elif config.period:
        parser = PeriodParser(config.raw_dir, config.user_record_file, config.item_record_file,
                              config.T, config.P, 'record')
        parser.parse_record(flags.user_record_file, flags.item_record_file,
                            flags.user_length_file, flags.item_length_file)
        task = 'period'
        # num_user, num_item = period_parser.NU, period_parser.NI
        # parse_period_set(config.raw_dir, config.train_file, config.T, config.P, num_user, num_item, flags.train_file,
        #                  'train')
        # parse_period_set(config.raw_dir, config.valid_file, config.T, config.P, num_user, num_item, flags.valid_file,
        #                  'valid')
        # parse_period_set(config.raw_dir, config.test_file, config.T, config.P, num_user, num_item, flags.test_file,
        #                  'test')
    else:
        parser = BasicParser(config.raw_dir, config.user_record_file, config.item_record_file,
                             config.T, config.P, 'record')
        parser.parse_record(flags.user_record_file, flags.item_record_file,
                            flags.user_length_file, flags.item_length_file)
        task = 'static'
        # parse_set(config.raw_dir, config.train_file, config.T, flags.train_file, 'train')
        # parse_set(config.raw_dir, config.valid_file, config.T, flags.valid_file, 'valid')
        # parse_set(config.raw_dir, config.test_file, config.T, flags.test_file, 'test')

    parser.parse_set(config.raw_dir, config.train_file, flags.train_file, 'train', task)
    parser.parse_set(config.raw_dir, config.valid_file, flags.valid_file, 'valid', task)
    parser.parse_set(config.raw_dir, config.test_file, flags.test_file, 'test', task)
