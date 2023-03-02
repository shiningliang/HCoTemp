import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import multiprocessing
import pickle as pkl
import random
import ujson as json

plt.switch_backend('agg')


def func_year(date):
    dl = date.split('-')
    return dl[0]


def func_month(date):
    dl = date.split('-')
    return dl[1]


def get_records(group, sort_id):
    if sort_id == 'userID':
        sid = group.iloc[0, 1]
    else:
        sid = group.iloc[0, 0]
    months, ratings = [], []
    j_year = group.groupby('year')
    for _, yj in j_year:
        j_month = yj.groupby('month')
        for _, mj in j_month:
            id_list = mj[sort_id].tolist()
            months.append(id_list)
            rating_list = mj['rating'].tolist()
            ratings.append({k: v for k, v in zip(id_list, rating_list)})
    return sid, months, ratings


def stat_len(samples, sample_type):
    outer_len, inner_len = [], []
    for sample in samples.values():
        outer_len.append(len(sample))
        for rec in sample:
            inner_len.append(len(rec))

    show_len(inner_len, sample_type + "_inner")
    show_len(outer_len, sample_type + "_outer")


def show_len(seq, seq_type):
    print('Seq len info: ', seq_type)
    seq_len = np.asarray(seq)
    idx = np.arange(0, len(seq_len), dtype=np.int32)
    print(stats.describe(seq_len))
    plt.figure(figsize=(16, 9))
    plt.subplot(121)
    plt.plot(idx[:], seq_len[:], 'ro')
    plt.grid(True)
    plt.xlabel('index')
    plt.ylabel(seq_type)
    plt.title('Scatter Plot')

    plt.subplot(122)
    plt.hist(seq_len, bins=5, label=['seq_len'])
    plt.grid(True)
    plt.xlabel(seq_type)
    plt.ylabel('freq')
    plt.title('Histogram')
    # plt.show()
    plt.savefig("./" + seq_type + ".jpg", format='jpg')


def read_file(file_path):
    raw_table = pd.read_csv(file_path, sep=',', header=None,
                            names=['userID', 'movieID', 'catID', 'reviewID', 'rating', 'date'])
    del raw_table['catID']
    del raw_table['reviewID']

    raw_table['year'] = raw_table.apply(lambda x: func_year(x.date), axis=1)
    raw_table['month'] = raw_table.apply(lambda x: func_month(x.date), axis=1)
    del raw_table['date']
    u_table = raw_table.sort_values(by=['userID', 'year', 'month', 'movieID'])
    u_table.reset_index(drop=True, inplace=True)
    u_group = u_table.groupby('userID')
    print('Num of raw users - {}'.format(len(u_group.count())))

    i_table = raw_table.sort_values(by=['movieID', 'year', 'month', 'userID'])
    i_table.reset_index(drop=True, inplace=True)
    i_group = i_table.groupby('movieID')
    print('Num of raw items - {}'.format(len(i_group.count())))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = []
    for _, ig in i_group:
        results.append(pool.apply_async(get_records, (ig, 'userID',)))
    pool.close()
    pool.join()
    item_records = {res.get()[0]: res.get()[1] for res in results}
    item_ratings = {res.get()[0]: res.get()[2] for res in results}
    item_records = dict(sorted(item_records.items(), key=lambda x: x[0]))
    item_ratings = dict(sorted(item_ratings.items(), key=lambda x: x[0]))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = []
    for _, ug in u_group:
        results.append(pool.apply_async(get_records, (ug, 'movieID',)))
    pool.close()
    pool.join()
    user_records = {res.get()[0]: res.get()[1] for res in results}
    user_ratings = {res.get()[0]: res.get()[2] for res in results}

    return user_records, user_ratings, item_records, item_ratings


def remove(samples, ratings, remove_list):
    old_item = samples.keys()
    for sid, month in samples.items():
        samples[sid] = []
        scores = ratings[sid]
        ratings[sid] = []
        for idx, rec in enumerate(month):
            ids = set(rec).intersection(set(remove_list))  # 在remove_list中的id
            rec = list(set(rec).difference(ids))  # 除此之外剩余的id
            for id in ids:
                scores[idx].pop(id)
            if len(rec) > 0:
                samples[sid].append(rec)
            if len(scores[idx]) > 0:
                ratings[sid].append(scores[idx])
    samples = {k: v for k, v in samples.items() if len(v) >= 4}
    ratings = {k: v for k, v in ratings.items() if len(v) >= 4}
    remove_new = list(set(old_item) - set(samples.keys()))

    return samples, ratings, remove_new


def clean_low_len(u_samples, i_samples, u_ratings, i_ratings):
    old_user = u_samples.keys()
    u_samples = {k: v for k, v in u_samples.items() if len(v) >= 4}
    u_remove = list(set(old_user) - set(u_samples.keys()))

    while len(u_remove) > 0:
        i_samples, i_ratings, i_remove = remove(i_samples, i_ratings, u_remove)
        u_samples, u_ratings, u_remove = remove(u_samples, u_ratings, i_remove)

    print('Num of filtered users - {}'.format(len(u_samples)))
    print('Num of filtered items - {}'.format(len(i_samples)))

    return u_samples, i_samples, u_ratings, i_ratings


def filer_records(u_recs, i_recs, u_ratings, i_ratings):
    u_samples, i_samples, u_ratings, i_ratings = clean_low_len(u_recs, i_recs, u_ratings, i_ratings)

    u_list = list(u_samples.keys())
    u_map = {uid: idx + 1 for idx, uid in enumerate(u_list)}
    i_list = list(i_samples.keys())
    i_map = {iid: idx + 1 for idx, iid in enumerate(i_list)}

    u_rec_sort = {u_map[k]: [[i_map[rec] for rec in month] for month in v] for k, v in u_samples.items()}
    u_rat_sort = {u_map[k]: [list(rating.values()) for rating in v] for k, v in u_ratings.items()}
    i_rec_sort = {i_map[k]: [[u_map[rec] for rec in month] for month in v] for k, v in i_samples.items()}
    i_rat_sort = {i_map[k]: [list(rating.values()) for rating in v] for k, v in i_ratings.items()}

    stat_len(u_rec_sort, 'user')
    stat_len(i_rec_sort, 'item')

    with open('u_rec_sort.pkl', 'wb') as fu:
        pkl.dump(u_rec_sort, fu)
    fu.close()
    with open('i_rec_sort.pkl', 'wb') as fi:
        pkl.dump(i_rec_sort, fi)
    fi.close()
    with open('u_rat_sort.pkl', 'wb') as fu:
        pkl.dump(u_rat_sort, fu)
    fu.close()
    with open('i_rat_sort.pkl', 'wb') as fi:
        pkl.dump(i_rat_sort, fi)
    fi.close()


def remove_uid_in_irecs(recs, uid):
    pop_list = []
    for i in range(len(recs)):
        if uid in recs[i]:
            recs[i].remove(uid)
            if len(recs[i]) == 0:
                pop_list.append(i)

    if len(pop_list) > 0:
        for _, idx in enumerate(pop_list):
            recs.pop(idx)


if __name__ == '__main__':
    # path = './data/raw_data/movie-ratings.txt'
    # user_records, user_ratings, item_records, item_ratings = read_file(path)
    # filer_records(user_records, item_records, user_ratings, item_ratings)

    # stat_len(user_records, 'user')
    # stat_len(item_records, 'item')

    with open('u_rec.pkl', 'rb') as fu:
        u_recs = pkl.load(fu)
    fu.close()
    with open('i_rec.pkl', 'rb') as fi:
        i_recs = pkl.load(fi)
    fi.close()
    with open('u_rat.pkl', 'rb') as fu:
        u_rats = pkl.load(fu)
    fu.close()
    with open('i_rat.pkl', 'rb') as fi:
        i_rats = pkl.load(fi)
    fi.close()

    filer_records(u_recs, i_recs, u_rats, i_rats)

    num_user = len(u_recs)
    user_list = set([x for x in range(num_user)])
    num_item = len(i_recs)
    item_list = set([y for y in range(1, num_item + 1)])

    train_uids, train_iids, train_labels = [], [], []
    dev_uids, dev_iids, dev_labels = [], [], []
    test_uids, test_iids, test_labels = [], [], []
    for uid, v in u_recs.items():
        num_train = len(v[-3])
        num_dev = len(v[-2])
        num_test = len(v[-1])
        train_uids.extend([uid] * num_train)
        train_iids.extend(v[-3])
        # train_labels.extend([1] * num_train)
        train_labels.extend([round(rat * 0.2, 2) for rat in u_rats[uid][-3]])
        dev_uids.extend([uid] * num_dev)
        dev_iids.extend(v[-2])
        # dev_labels.extend([1] * num_dev)
        dev_labels.extend([round(rat * 0.2, 2) for rat in u_rats[uid][-2]])
        test_uids.extend([uid] * num_test)
        test_iids.extend(v[-1])
        # test_labels.extend([1] * num_test)
        test_labels.extend([round(rat * 0.2, 2) for rat in u_rats[uid][-1]])

        for iid in v[-1]:
            remove_uid_in_irecs(i_recs[iid], uid)

        for iid in v[-2]:
            remove_uid_in_irecs(i_recs[iid], uid)

        rec_list = []
        for month in v:
            rec_list.extend(month)
        rec_list = set(rec_list)
        neg_list = item_list - rec_list
        neg_ids = random.sample(neg_list, num_train)
        train_uids.extend([uid] * num_train)
        train_iids.extend(neg_ids)
        train_labels.extend([0] * num_train)

        neg_list = neg_list - set(neg_ids)
        neg_ids = random.sample(neg_list, num_dev)
        dev_uids.extend([uid] * num_dev)
        dev_iids.extend(neg_ids)
        dev_labels.extend([0] * num_dev)

        neg_list = neg_list - set(neg_ids)
        neg_ids = random.sample(neg_list, num_test)
        test_uids.extend([uid] * num_test)
        test_iids.extend(neg_ids)
        test_labels.extend([0] * num_test)

    for k, v in i_recs.items():
        if len(v) == 0:
            print(k)

    print(len(train_labels), len(dev_labels), len(test_labels))


    def save_set(uids, iids, labels, settype):
        df = pd.DataFrame({'uids': uids, 'iids': iids, 'labels': labels})
        df.to_csv('./data/raw_data/' + settype + '.csv', sep=',', index=False)


    def save_record(rectype, recs):
        with open('./data/raw_data/' + rectype + '_record.json', 'w') as f:
            for k, v in recs.items():
                tmp_str = json.dumps({k: v})
                f.write(tmp_str + '\n')


    save_set(train_uids, train_iids, train_labels, 'train')
    save_set(dev_uids, dev_iids, dev_labels, 'dev')
    save_set(test_uids, test_iids, test_labels, 'test')

    save_record('user', u_recs)
    save_record('item', i_recs)

    print('hello world')
