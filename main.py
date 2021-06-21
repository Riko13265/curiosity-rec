
from DRAFT import load, save, new_model, get_data, prof
from INPUT import input_csv_flixster, input_train_ratio, input_kernelmf_01, input_rs

from curiosity import curiosity_scoring, indexed_by_iu

# TODO: Re-train model

MODEL_PATH = './_model/___model02.pickle'

# index_train, model = new_model(input_csv_flixster, input_train_ratio, input_kernelmf_01)
# save(MODEL_PATH, (index_train, model))

index_train, model = load(MODEL_PATH)
ratings_train, ratings_test, trusts, p_ratings = get_data(input_csv_flixster, index_train, model)

scorings = curiosity_scoring(
    ratings_train,
    p_ratings,
    trusts,
    input_rs
)

# # NOTE: Create test user pool from test data
# # NOTE: Only users that got valid test data and pools can get recommended
#
# def asd(uid):
#     print(scorings.item_pool(uid))
#
# def asdf(uid):
#     for i in scorings.item_pool(uid):
#         print(scorings.ss(uid, i))
#
# prof(lambda: asdf(8), f'ss{8}.pstat')
# prof(lambda: asdf(11), f'ss{11}.pstat')
#
# prof(lambda: asd(8), f'spool{8}.pstat')
# prof(lambda: asd(11), f'spool{11}.pstat2')
#
# scorings.item_pool(8).to_list()
