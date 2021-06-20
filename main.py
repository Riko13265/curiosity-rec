import pickle

from DRAFT import from_raw_to_datas, random_partition, mf_model, build_p_rating, indexing
from INPUT import input_flixster, input_train_ratio, input_kernelmf_01, input_rs

from curiosity import curiosity_scoring

# data_rating, data_trust = from_raw_to_datas(input_flixster)
#
# index_train = random_partition(input_train_ratio, data_rating.index)
#
# data_rating_train = data_rating.loc[index_train]
# data_rating_test = data_rating.drop(index_train)
#
# model = mf_model(input_kernelmf_01, data_rating_train)
#
# with open('./___model01.pickle', 'wb') as handle:
#     pickle.dump(
#         (data_rating, data_trust,
#          index_train,
#          data_rating_train, data_rating_test,
#          model),
#         handle)

with open('./___model01.pickle', 'rb') as handle:
    (data_rating, data_trust,
     index_train,
     data_rating_train, data_rating_test,
     model) = pickle.load(handle)

data_p_rating = build_p_rating(model, data_rating)

ratings, ratings_iu, trusts, p_ratings = \
    indexing(data_rating_train, data_trust, data_p_rating)

scorings = curiosity_scoring(
    lambda u: ratings.loc[u].index,
    lambda i: ratings_iu.loc[i].index,
    lambda u, i: ratings.loc[u, i][0],
    lambda u, i: p_ratings.loc[u, i][0],
    lambda u: trusts.loc[u].index,
    input_rs
)

# NOTE: Create test user pool from test data
# NOTE: Only users that got valid ss, us, cs pools can get recommended



len(scorings.ss_pool(8))

scorings.ss_pool(8)
scorings.ss(8, 5124)
scorings.us(8, 55304)

