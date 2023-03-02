import json
import os
import sys
from collections import Counter, defaultdict

import torch
import dgl
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=8)


# print(__doc__)
#
# # 构建数据集
# rnd = np.random.RandomState(42)
# X = rnd.uniform(-3, 3, size=100)
# y = np.sin(X) + rnd.normal(size=len(X)) / 3
# X = X.reshape(-1, 1)
#
# # 用KBinsDiscretizer转换数据集
# enc = KBinsDiscretizer(n_bins=10, encode='onehot')
# X_binned = enc.fit_transform(X)
# X_binned = X_binned.todense()
#
# print("hello")


def parse(path):
    i = 0
    df = {}
    with open(path, "r") as f:
        for line in f:
            df[i] = json.loads(line)
            i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def extract_category(i, index):
    try:
        cate = i[index]
    except Exception:
        cate = "Unknown"
    return cate


def merge_category(a, b):
    return str(a + "##" + b)


def extract_brand(i):
    if i == '':
        brand = 'Unknown'
    else:
        if i.find(r'by') == 0:
            brand = i[2:].replace('\n', '').replace(' ', '')
        else:
            brand = i
    return brand


def extract_price(i):
    try:
        price = float(i[1:])
    except Exception:
        price = np.nan
    return price


def extract_rank(l, rank_set, rank_max):
    try:
        result_list = []
        for item in l:
            item_spl = item.split(' ')
            rank_num = int(item_spl[0][2:].replace(',', ''))
            rank_name = item[item.find("in ") + 3:]

            rank_set.add(rank_name)
            if rank_name not in rank_max:
                rank_max[rank_name] = rank_num
            if rank_max[rank_name] < rank_num:
                rank_max[rank_name] = rank_num
            result_list.append((rank_name, rank_num))
        return result_list
    except Exception:
        return []


def extract_rank_cd(l):
    try:
        result_list = []
        if type(l) == str:
            item_spl = l.split(' ')
            rank_num = int(item_spl[0].replace(',', ''))
            rank_name = l[l.find("in ") + 3:].replace(' (', '').strip()
            result_list.append((rank_name, rank_num))
        else:
            item_spl = l[0].split(' ')
            rank_num = int(item_spl[0][2:].replace(',', ''))
            rank_name = l[l.find("in ") + 3:]
            result_list.append((rank_name, rank_num))
        return result_list
    except Exception:
        return []


def gen_main_rank(x):
    if len(x) == 0:
        return np.nan
    else:
        return int(x[0][1])


def gen_sub_rank(x):
    if len(x) == 0 or len(x) == 1:
        return np.nan
    else:
        return int(x[1][1])


def build_metapath(df):
    # 使用二三级类目
    df['category2'] = df['category'].parallel_apply(extract_category, args=(1,))
    df['category3'] = df['category'].parallel_apply(extract_category, args=(2,))
    # 合并二三级类目
    df['category23'] = df.parallel_apply(lambda x: merge_category(x.category2, x.category3), axis=1)

    # 提取brand
    df['brand'] = df['brand'].parallel_apply(extract_brand)

    percent = 0.5  # 保留类目比例
    brand_lst = df['brand'].values
    brand_cnt = Counter(brand_lst)
    if "Unknown" in brand_cnt:
        brand_cnt.pop("Unknown")
    top_brand_num = int(len(brand_cnt) * 0.8)
    top_brand_lst = brand_cnt.most_common(top_brand_num)
    print("Brand Top {}% coverage - {}".format(0.8 * 100, sum([x[1] for x in top_brand_lst]) / sum(brand_cnt.values())))
    brand = [x[0] for x in top_brand_lst]  # 1399 in Games
    brand_map = defaultdict(int)
    brand_map.update([(b, i + 1) for i, b in enumerate(brand)])
    brand_num = len(brand_map)

    cate_lst = df['category23'].values
    cate_cnt = Counter(cate_lst)
    c2_lst = df['category2'].values
    c2_cnt = Counter(c2_lst)
    c3_lst = df['category3'].values
    c3_cnt = Counter(c3_lst)
    if "Unknown##Unknown" in cate_cnt:
        cate_cnt.pop("Unknown##Unknown")
    top_cate_num = int(len(cate_cnt) * percent)
    top_cate_lst = cate_cnt.most_common(top_cate_num)
    print("Category Top {}% coverage - {}".format(percent * 100, sum([x[1] for x in top_cate_lst]) / sum(cate_cnt.values())))
    cate = [x[0] for x in top_cate_lst]
    cate_map = defaultdict(int)
    cate_map.update([(c, i + 1) for i, c in enumerate(cate)])
    cate_num = len(cate_map)
    item_num = df['asin'].max()

    # TODO 以上只根据category建图，需要再加入根据brand属性建图

    g_c = np.zeros(shape=(item_num + 1, cate_num + 1))
    g_b = np.zeros(shape=(item_num + 1, brand_num + 1))
    for _, row in df.iterrows():
        asin = row['asin']
        cate_id = cate_map.get(row['category23'])
        brand_id = brand_map.get(row['brand'])
        g_b[asin][brand_id] = 1
        g_c[asin][cate_id] = 1
    g_c = scipy.sparse.csc_matrix(g_c)
    g_b = scipy.sparse.csc_matrix(g_b)

    hg = dgl.heterograph({
        ('item', 'ic', 'category'): g_c.nonzero(),
        ('category', 'ci', 'item'): g_c.transpose().nonzero(),
        ('item', 'ib', 'brand'): g_b.nonzero(),
        ('brand', 'bi', 'item'): g_b.transpose().nonzero(),
    })

    print("Generate hg Done!")

    return hg, df


def build_features(df, dataset_name):
    feature_file = "{}_graph_features.bin".format(dataset_name)
    if os.path.exists(feature_file):
        hg_features = torch.load(feature_file)
    else:
        # 提取rank
        print("Clean property")
        rank_set = set()
        rank_max = {}
        df.sort_values(by=['asin'], inplace=True)
        item_num = df['asin'].max()
        if dataset_name in ['AM_Games']:
            df['rank'] = df['rank'].parallel_apply(extract_rank, args=(rank_set, rank_max,))
        else:
            df['rank'] = df['rank'].parallel_apply(extract_rank_cd)
        # 提取price
        df['price'] = df['price'].parallel_apply(extract_price)

        df['rank_main'] = df['rank'].parallel_apply(gen_main_rank)
        rm_lst = df['rank_main'].values
        rm_cnt = Counter(rm_lst)
        if dataset_name in ['AM_Games']:
            df['rank_sub'] = df['rank'].parallel_apply(gen_sub_rank)
            rs_lst = df['rank_sub'].values
            rs_cnt = Counter(rs_lst)
        df['category2'] = LabelEncoder().fit_transform(df['category2'])
        df['category3'] = LabelEncoder().fit_transform(df['category3'])

        print("Fill missing value")
        # imp = IterativeImputer(max_iter=100, random_state=0)
        imp = KNNImputer(n_neighbors=10, weights="uniform")

        if dataset_name in ['AM_Games']:
            df[['price', 'rank_main', 'rank_sub', 'category2', 'category3']] = imp.fit_transform(
                df[['price', 'rank_main', 'rank_sub', 'category2', 'category3']])
            enc = KBinsDiscretizer(n_bins=[16, 16, 32], encode='onehot')
            X_binned = enc.fit_transform(df[['price', 'rank_main', 'rank_sub']])
        else:
            df[['price', 'rank_main', 'category2', 'category3']] = imp.fit_transform(
                df[['price', 'rank_main', 'category2', 'category3']])
            enc = KBinsDiscretizer(n_bins=[32, 32], encode='onehot')
            X_binned = enc.fit_transform(df[['price', 'rank_main']])
        X_binned = X_binned.todense()

        print("Fill feature matrix")
        h_r_p = np.random.randn(item_num + 1, 64)
        corpus_dict = {}
        for idx, x in df.iterrows():
            h_r_p[x.asin] = X_binned[idx]
            # if len(x.description) > 0:
            #     corpus_dict[x.asin] = x.title + " ## " + x.description[0]
            # else:
            #     corpus_dict[x.asin] = x.title + " ## "

        # corpus = list(corpus_dict.values())
        # tfidf = TfidfVectorizer()
        # tf = tfidf.fit_transform(corpus)

        # rank_map = dict([(rank, i) for i, rank in enumerate(rank_set)])
        # rank_num = len(rank_map)
        # item_num = df['asin'].max()
        # h_r_p = -np.ones(shape=(item_num + 1, rank_num + 1))
        # for _, row in df.iterrows():
        #     asin = row['asin']
        #     price = row['price']
        #     for rank_item in row['rank']:
        #         rank_id = rank_map[rank_item[0]]
        #         h_r_p[asin][rank_id] = rank_item[1]
        #     h_r_p[asin][rank_num] = price

        # sentences = df['title'].tolist()
        # embedding_file = './{}_sentence_embeddings.npy'.format(dataset_name)
        # if os.path.exists(embedding_file):
        #     sentence_embeddings = np.load(embedding_file)
        # else:
        #     print('Get Sentence Embedding')
        #     model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
        #     sentence_embeddings = model.encode(sentences)
        #     with open(embedding_file, 'wb') as f:
        #         np.save(f, sentence_embeddings)

        # TODO 以上只实现title的sentence embedding做节点特征，造成显存不足，需要加入price salesRank做特征
        # TODO title以bag-of-words表示若仍显存不足，可不用

        # asins = df['asin'].tolist()

        # item_num = df['asin'].max()
        # feature_num = sentence_embeddings.shape[1]
        # hg_features = np.zeros(shape=(item_num + 1, feature_num))
        # for asin, sentence_embedding in zip(asins, sentence_embeddings):
        #     hg_features[asin, :] = sentence_embedding
        # hg_features = torch.FloatTensor(hg_features)
        h_r_p = torch.FloatTensor(h_r_p)

        # hg_features = torch.cat([hg_features, h_r_p], dim=1)
        # torch.save(hg_features, feature_file)
        torch.save(h_r_p, feature_file)
        hg_features = h_r_p

    return hg_features


if __name__ == '__main__':
    task = sys.argv[1]
    df = parse('./data/raw_data/{}/new_metadata2.json'.format(task))
    print(df)
    hg, df = build_metapath(df)
    features = build_features(df, task)
    print(hg)
