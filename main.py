
from DRAFT import load, save, new_model, get_data, prof
from INPUT import input_csv_flixster, input_train_ratio, input_kernelmf_01, input_rs

from curiosity import curiosity_scoring

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

# NOTE: Create test user pool from test data
# NOTE: Only users that got valid test data and pools can get recommended

def asdf(uid):
    for i in scorings.item_pool(uid):
        scorings.ss(uid, i)

prof(lambda: asdf(8), f'ss{8}.pstat')
prof(lambda: asdf(11), f'ss{11}.pstat')