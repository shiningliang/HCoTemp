#!/usr/bin/env python
# encoding: utf-8
"""
@author: zhongli
@time: 2021/4/21 19:11
"""
import json
import os
import sys

import dgl
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from sentence_transformers import SentenceTransformer


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


# 构建异质图
def build_metapath(df):
    # 使用二三级类目
    df['category2'] = df['category'].apply(extract_category, args=(1,))
    df['category3'] = df['category'].apply(extract_category, args=(2,))

    cate = set(np.append(df['category2'].unique(), df['category3'].unique()))
    cate_map = dict([(cate, i) for i, cate in enumerate(cate)])
    cate_num = len(cate_map)
    item_num = df['asin'].max()

    # TODO 以上只根据category建图，需要再加入根据brand属性建图

    g = np.zeros(shape=(item_num + 1, cate_num))
    for _, row in df.iterrows():
        asin = row['asin']
        cate2_id = cate_map.get(row['category2'])
        cate3_id = cate_map.get(row['category3'])
        g[asin][cate2_id] = 1
        g[asin][cate3_id] = 1
    g = scipy.sparse.csc_matrix(g)

    hg = dgl.heterograph({
        ('item', 'ic', 'category'): g.nonzero(),
        ('category', 'ci', 'item'): g.transpose().nonzero(),
    })

    return hg


def build_features(df, dataset_name):
    feature_file = "{}_graph_features.bin".format(dataset_name)
    if os.path.exists(feature_file):
        hg_features = torch.load(feature_file)
    else:
        sentences = df['title'].tolist()
        embedding_file = './{}_sentence_embeddings.npy'.format(dataset_name)
        if os.path.exists(embedding_file):
            sentence_embeddings = np.load(embedding_file)
        else:
            print('Get Sentence Embedding')
            model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
            sentence_embeddings = model.encode(sentences)
            with open(embedding_file, 'wb') as f:
                np.save(f, sentence_embeddings)

        # TODO 以上只实现title的sentence embedding做节点特征，造成显存不足，需要加入price salesRank做特征
        # TODO title以bag-of-words表示若仍显存不足，可不用

        asins = df['asin'].tolist()

        item_num = df['asin'].max()
        feature_num = sentence_embeddings.shape[1]
        hg_features = np.zeros(shape=(item_num + 1, feature_num))
        for asin, sentence_embedding in zip(asins, sentence_embeddings):
            hg_features[asin, :] = sentence_embedding
        hg_features = torch.FloatTensor(hg_features)
        torch.save(hg_features, feature_file)

    return hg_features


if __name__ == '__main__':
    task = sys.argv[1]
    df = parse('./data/{}/new_metadata2.json'.format(task))
    print(df)
    hg = build_metapath(df)
    features = build_features(df, task)
    print(hg)
