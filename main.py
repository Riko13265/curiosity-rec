
from DRAFT import load, save, new_model, get_data, from_raw_to_datas, recommendation, rank_in_test
from INPUT import input_csv_flixster, input_rs

from curiosity import curiosity_scoring, cf_scoring

# TODO: Re-train model

MODEL_PATH = './_model/___model03.pickle'

data_rating, data_trust = from_raw_to_datas(input_csv_flixster)

# index_train, model = new_model(input_csv_flixster, input_train_ratio, input_kernelmf_01)
# save(MODEL_PATH, (index_train, model))

index_train, model = load(MODEL_PATH)

ratings_train, ratings_test, trusts, p_ratings = get_data(data_rating, data_trust, index_train, model)

all_users = data_rating['user_id'].unique()
mvp_users = data_rating.groupby('user_id').count().loc[lambda x: x['item_id'] >= 10].index
items = data_rating['item_id'].unique()

curiosity = curiosity_scoring(
    ratings_train,
    p_ratings,
    trusts,
    input_rs
)

cf = cf_scoring(
    items,
    ratings_train,
    model
)

# NOTE: Create test user pool from test data
# NOTE: Only users that got valid test data and pools can get recommended
# FIXME: ^^^ Using try-except now

rec = recommendation(
    curiosity, cf)

test = rank_in_test(
    ratings_test)

def precision(u, n):
    return len(rec(u)[:n].loc[lambda x: x.isin(test(u))]) / n

asdfgh = [precision(u, 10) for u in mvp_users[:100]]

sum(asdfgh)/len(asdfgh)