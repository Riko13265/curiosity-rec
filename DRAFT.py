from random import shuffle
from matrix_factorization import KernelMF
from pandas import read_csv


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


def mf_model(input_kernelmf, data_rating_train):
    model = KernelMF(**input_kernelmf)
    model.fit(data_rating_train[['user_id', 'item_id']],
              data_rating_train[['rating']])
    return model


def build_p_rating(model, data_rating):
    return data_rating.drop("rating", "columns") \
        .assign(rating=model.predict(data_rating[['user_id', 'item_id']]))


def indexing(data_rating, data_trust, data_p_rating):
    rated_user_set = set(data_rating['user_id'].to_list())
    # trusting_user_set = set(data_trust['u'].to_list())

    ratings = data_rating\
        .set_index(['user_id', 'item_id']).sort_index()
    ratings_iu = data_rating\
        .set_index(['item_id', 'user_id']).sort_index()
    trusts = data_trust[data_trust['u'].isin(rated_user_set) &
                        data_trust['v'].isin(rated_user_set)] \
        .set_index(['u', 'v']).sort_index()
    p_ratings = data_p_rating\
        .set_index(['user_id', 'item_id']).sort_index()
    return ratings, ratings_iu, trusts, p_ratings
