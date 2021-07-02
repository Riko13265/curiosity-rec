import pickle
from random import shuffle
from typing import NamedTuple

from matrix_factorization import KernelMF
from pandas import read_csv

import cProfile

from borda_count import borda_count

pr = cProfile.Profile()


def prof(func, name):
    pr = cProfile.Profile()
    pr.enable()
    func()
    pr.disable()
    pr.dump_stats(f'./_pstat/{name}')


def save(pickle_path, thing):
    with open(pickle_path, 'wb') as handle:
        pickle.dump(thing, handle)


def load(pickle_path):
    with open(pickle_path, 'rb') as handle:
        return pickle.load(handle)


def from_raw_to_datas(raw_info):
    data_rating = read_csv(raw_info['csv_rating_path'],
                           **raw_info['csv_rating_opts'])
    data_trust = read_csv(raw_info['csv_trust_path'],
                          **raw_info['csv_trust_opts'])
    return data_rating, data_trust


def random_partition(ratio, col):
    col_list = list(col)
    length = len(col)
    anchor = int(ratio * length)
    shuffle(col_list)
    return sorted(col_list[:anchor])


def mf(input_mf, data_rating_train):
    model = KernelMF(**input_mf)
    model.fit(data_rating_train[['user_id', 'item_id']],
              data_rating_train[['rating']])
    return model


def new_model(input_csv, input_train_ratio, input_mf_params):
    data_rating, _ = from_raw_to_datas(input_csv)
    index_train = random_partition(input_train_ratio, data_rating.index)
    data_rating_train = data_rating.loc[index_train]
    model = mf(input_mf_params, data_rating_train)
    return index_train, model


def build_p_rating(model, data_rating):
    return data_rating.drop("rating", "columns") \
        .assign(rating=model.predict(data_rating[['user_id', 'item_id']]))


def indexing(index_train, data_rating, data_trust, data_p_rating):
    rated_user_set = set(data_rating['user_id'].to_list())

    ratings_train = data_rating.loc[index_train] \
        .set_index(['user_id', 'item_id']).sort_index()
    ratings_test = data_rating.drop(index_train) \
        .set_index(['user_id', 'item_id']).sort_index()
    trusts = data_trust[data_trust['u'].isin(rated_user_set) &
                        data_trust['v'].isin(rated_user_set)] \
        .set_index(['u', 'v']).sort_index()
    p_ratings = data_p_rating \
        .set_index(['user_id', 'item_id']).sort_index()
    return ratings_train, ratings_test, trusts, p_ratings


def get_data(data_rating, data_trust, index_train, model):
    # data_rating, data_trust = from_raw_to_datas(input_csv)
    data_p_rating = build_p_rating(model, data_rating)
    return indexing(index_train, data_rating, data_trust, data_p_rating)


import pandas as pd


def recommendation(curiosity, cf):
    def rec(u):
        try:
            return borda_count(curiosity.score(u), cf.score(u), 0.5)
        except KeyError:
            return pd.Series([], dtype=int).rename('item_id')
    return rec


def rank_in_test(ratings_test):
    def test(u):
        try:
            return ratings_test.loc[u].reset_index()['item_id']
        except KeyError:
            return pd.Series([], dtype=int).rename('item_id')
    return test