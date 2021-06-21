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

prof(lambda: print(scorings.ss(11)), f'v2_ss_{11}.pstat')
prof(lambda: print(scorings.us(11)), f'v2_us_{11}.pstat')
prof(lambda: print(scorings.cs(11)), f'v2_cs_{11}.pstat')

prof(lambda: print(scorings.ss(8)), f'v2_ss_{8}.pstat')
prof(lambda: print(scorings.us(8)), f'v2_us_{8}.pstat')
prof(lambda: print(scorings.cs(8)), f'v2_cs_{8}.pstat')
