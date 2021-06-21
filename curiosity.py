import math
from functools import cache
from typing import NamedTuple, Callable, List

import pandas as pd

IdToSeries = Callable[[int], pd.Series]
IdToIndex = Callable[[int], pd.Index]


class CuriosityScoring(NamedTuple):
    ss: IdToSeries
    us: IdToSeries
    cs: IdToSeries
    item_pool: IdToIndex


def indexed_by_iu(df):
    return df.reorder_levels(['item_id', 'user_id']).sort_index()


def safe_loc_index(df, index):
    return df.loc[[index]].index.droplevel(0)


def filter_index(index, predicate):
    frame = index.to_frame().iloc[:, 0]
    return frame.loc[frame.apply(predicate)].index


def xlogx(x):
    if x == 0.:
        return 0.
    else:
        return x * math.log(x)


def curiosity_scoring(
        ratings: pd.DataFrame,
        p_ratings: pd.DataFrame,
        trusts: pd.DataFrame,
        rs: List[int]
):
    r_sub_pr = (ratings - p_ratings).dropna()
    s_ui = r_sub_pr[r_sub_pr['rating'] > 0.]
    s_iu = indexed_by_iu(s_ui)

    @cache
    def s_uiv(u):
        sui = s_ui.rename_axis(index={'user_id': 'u'})
        siv = s_iu.rename_axis(index={'user_id': 'v'})
        return sui.loc[u].merge(siv, left_index=True, right_index=True) \
            .reorder_levels(['v', 'item_id']) \
            .sort_index().loc[trusts.loc[u].index]

    def correlated_friends(u):
        return s_uiv(u).index.droplevel(1).unique()

    r_m = max(rs) - min(rs)

    def sc(u):
        suivu = s_uiv(u)
        diff = abs(suivu['rating_x'] - suivu['rating_y'])
        return (1. - diff.groupby('v').sum() / diff.groupby('v').agg('count') / r_m).to_frame(name='sc')

    def ss(u):
        siv_sc = sc(u).merge(
            s_ui.rename_axis(index={'user_id': 'v'}),
            left_index=True, right_index=True
        )
        s_sc = (siv_sc['rating'] * siv_sc['sc'])
        return s_sc.groupby('item_id').sum() / s_sc.groupby('item_id').agg('count')

    @cache
    def ss_pool(u):
        return correlated_friends(u).rename('v').join(
            s_ui.index.rename(['v', 'item_id']),
            how='inner'
        ).droplevel(0).unique().sort_values()

    @cache
    def v_ir(u):
        return trusts.loc[u].merge(
            ratings.rename_axis(index={'user_id': 'v'}), left_index=True, right_index=True) \
            .reset_index().set_index(['item_id', 'rating']).sort_index()

    def c_ir(u):
        return v_ir(u).groupby(['item_id', 'rating']).agg('count')

    @cache
    def c_i(u):
        return v_ir(u).groupby(['item_id']).agg('count')

    def p_ir(u):
        return c_ir(u) / c_i(u)

    def se_i(u):
        return - p_ir(u).applymap(xlogx).groupby(['item_id']).sum()

    r = len(rs)

    def ds_i(u):
        return r / (c_i(u) + r)

    def us(u):
        return ((1. - ds_i(u)) * se_i(u)).loc[ss_pool(u)]['v']

    # NOTE: CS

    @cache
    def r_iv(u):
        return trusts.loc[u].merge(
            ratings.rename_axis(index={'user_id': 'v'}), left_index=True, right_index=True) \
            .reorder_levels(['item_id', 'v']).sort_index()

    def r_avg_i(u):
        return r_iv(u).groupby('item_id').mean()

    def cs(u):
        return (r_iv(u) - r_avg_i(u)) \
            .applymap(lambda x: pow(x, 2)) \
            .groupby('item_id').mean() \
            .applymap(math.sqrt) \
            .loc[ss_pool(u)]['rating']

    return CuriosityScoring(
        ss,
        us,
        cs,
        ss_pool
    )
