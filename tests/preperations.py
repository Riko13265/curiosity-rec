from DRAFT import load
from INPUT import input_csv_flixster, input_train_ratio, input_kernelmf_01
from operations import generate_new_vusers_from_mvp, train_new_model, group_recommendation_with_sc, evaluate

model_path = './_model/___model04.pickle'
vusers_path = './_vusers/___vusers01.pickle'

train_new_model(
    input_csv_flixster,
    input_train_ratio,
    input_kernelmf_01,
    save_path=model_path)

generate_new_vusers_from_mvp(
    input_csv_flixster,
    group_count=100,
    group_size=4,
    save_path=vusers_path)

index_train, model = load(model_path)
vusers = load(vusers_path)
settings = {
    'rec_count': 10,
    'vote_threshold': 1,
    'ratings_aggr': 'avg',
    'pratings_aggr': 'avg',
}

rec, rec_vuser, test = \
    group_recommendation_with_sc(
        input_csv_flixster,
        index_train, model,
        vusers, settings)

evaluate(vusers, rec, rec_vuser, test, settings)
