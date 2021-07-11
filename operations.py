from DRAFT import save, new_model, from_raw_to_datas, generate_vusers


# TODO: Re-train model

def train_new_model(input_csv_source, input_train_ratio, input_kernelmf_params, save_path):
    # NOTE: Load raw data
    data_rating, _ = from_raw_to_datas(input_csv_source)
    # NOTE: Train new model
    index_train, model = new_model(
        data_rating, input_train_ratio, input_kernelmf_params)
    # NOTE: Save model to MODEL_PATH
    save(save_path, (index_train, model))


def generate_new_vusers_from_mvp(input_csv_source, group_count, group_size, save_path):
    data_rating, data_trust = from_raw_to_datas(input_csv_source)
    mvp_users = data_rating.groupby('user_id').count().loc[lambda x: x['item_id'] >= 10].index
    vusers = generate_vusers(group_count, group_size, mvp_users)
    save(save_path, vusers)

import pandas as pd
from curiosity import curiosity_scoring, cf_scoring
from DRAFT import get_data, recommendation, rank_in_test

aggr_methods = {
    'avg': lambda x: x.sum() / x.count(),
    'min': lambda x: x.min(),
    'max': lambda x: x.max()
}


def group_recommendation_with_sc(input_csv_source,
                                 index_train, model,
                                 vusers,
                                 settings):
    # NOTE: Load model from MODEL_PATH
    # index_train, model = load(model_path)
    # NOTE: Load raw data
    data_rating, data_trust = from_raw_to_datas(input_csv_source)

    items = data_rating['item_id'].unique()

    # NOTE: Sampling and indexing
    ratings_train, ratings_test, trusts, p_ratings = get_data(data_rating, data_trust, index_train, model)

    curiosity = curiosity_scoring(
        ratings_train,
        p_ratings,
        trusts,
        input_csv_source['rs']
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
        curiosity.score, cf.score)

    # NOTE: Group/Virtual user tests

    # vusers = load(vusers_path)

    def vuser_aggr(ratings, aggr):
        vusers_iu_rating = vusers \
            .merge(ratings, left_index=True, right_index=True) \
            .reorder_levels(['vuser_id', 'item_id', 'user_id']).sort_index()

        vusers_i_rating_group = vusers_iu_rating.droplevel('user_id').groupby(['vuser_id', 'item_id'])
        vusers_i_rating_avg = aggr(vusers_i_rating_group)

        return pd.concat([
            vusers_i_rating_avg.rename_axis(index={'vuser_id': 'user_id'}),
            ratings
        ]).sort_index()

    # NOTE: Virtual user R
    # NOTE: It may seems like averaged ratings as vuser rating wont work in US,
    # NOTE: but since US involved no vuser, it would work as usual.

    # ratings_train_with_vusers = vuser_aggr(ratings_train, lambda x: x.sum() / x.count())
    # ratings_train_with_vusers = vuser_aggr(ratings_train, lambda x: x.min())
    # ratings_train_with_vusers = vuser_aggr(ratings_train, lambda x: x.max())
    ratings_train_with_vusers = vuser_aggr(ratings_train, aggr_methods[settings['ratings_aggr']])

    # NOTE: Virtual user PR
    # FIXME: PR was based on train items. In the case of virtual users,
    # FIXME: PR of every group member on every items should be provided.
    # FIXME: Using AVG now

    # p_ratings_with_vusers = vuser_aggr(p_ratings)
    data_vu_iu_rating = vusers \
        .merge(ratings_train, left_index=True, right_index=True) \
        .reset_index()

    data_vu_item = data_vu_iu_rating[['vuser_id', 'item_id']].drop_duplicates()
    data_vu_user = data_vu_iu_rating[['vuser_id', 'user_id']].drop_duplicates()

    prrrr = data_vu_user.merge(data_vu_item)
    prrrr.insert(2, 'rating', model.predict(prrrr))
    prrrr2 = prrrr.set_index(['vuser_id', 'item_id', 'user_id']) \
        .groupby(['vuser_id', 'item_id'])

    p_ratings_with_vusers = \
        pd.concat([
            aggr_methods[settings['pratings_aggr']](prrrr2).rename_axis(index={'vuser_id': 'user_id'}),
            p_ratings
        ]).sort_index()

    # NOTE: Virtual user trust
    # FIXME: Select trusts base on counting?
    # FIXME: Not many valid vusers with vote_threshold = 2 :( (~ 72/1000)
    # FIXME: Not a surprise with randomly-selected groups?

    vusers_uv = vusers.rename_axis(index={'user_id': 'u'}) \
        .merge(trusts, left_index=True, right_index=True) \
        .reorder_levels(['vuser_id', 'u', 'v']).sort_index().assign(count=1)

    vusers_v_count = vusers_uv.droplevel('u').groupby(['vuser_id', 'v']).count()

    vusers_v_voted = vusers_v_count.loc[vusers_v_count['count'] >= settings['vote_threshold']].index

    trusts_with_vusers = pd.concat([
        vusers_v_voted.to_frame().drop(['vuser_id', 'v'], axis=1),
        trusts
    ]).sort_index()

    curiosity_with_vuser = curiosity_scoring(
        ratings_train_with_vusers,
        p_ratings_with_vusers,
        trusts_with_vusers,
        input_csv_source['rs']
    )

    cf_with_vuser = cf_scoring(
        items,
        ratings_train_with_vusers,
        model
    )

    rec_vuser = recommendation(
        curiosity_with_vuser.score, cf_with_vuser.score)

    # NOTE: Evaluation

    test = rank_in_test(ratings_test)

    return rec, rec_vuser, test


def evaluate(vusers, rec, rec_vuser, test, settings):
    def precision(u, n):
        return len(rec(u)[:n].loc[lambda x: x.isin(test(u))]) / n

    def precision_vuser_as_of_individual(vu, n):
        users = vusers.loc[vu].index
        return [precision(u, n) for u in users]

    def precision_vuser(vu, n):
        rec_vu_n = rec_vuser(vu)[:n]
        users = vusers.loc[vu].index
        return [len(rec_vu_n.loc[lambda x: x.isin(test(u))]) / n for u in users]

    def avg(col):
        return sum(col) / len(col)

    avg_precision_vuser_as_of_individual = avg(
        [avg(precision_vuser_as_of_individual(vu, settings['rec_count']))
         for vu in vusers.index.get_level_values(0).unique()]
    )

    avg_precision_vuser = avg(
        [avg(precision_vuser(vu, settings['rec_count']))
         for vu in vusers.index.get_level_values(0).unique()]
    )