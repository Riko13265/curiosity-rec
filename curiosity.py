import math
from functools import cache
from typing import NamedTuple, Set, Callable, List

import pandas as pd

IdIdInt = Callable[[int, int], int]

IdIdFloat = Callable[[int, int], float]

IdToIdSet = Callable[[int], Set[int]]

IdToIndex = Callable[[int], pd.Index]


class CuriosityScoring(NamedTuple):
    ss: IdIdFloat
    us: IdIdFloat
    cs: IdIdFloat
    item_pool: IdToIdSet


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
    ratings_iu = indexed_by_iu(ratings)

    r_sub_pr = (ratings - p_ratings).dropna()

    s_ui = r_sub_pr[r_sub_pr['rating'] > 0.]
    s_u  = s_ui.index.droplevel(1).unique()
    s_iu = indexed_by_iu(s_ui)

    def rating(u, i):
        return ratings.loc[u, i][0]

    def users_rated(i):
        return safe_loc_index(ratings_iu, i)

    def friends(u):
        return safe_loc_index(trusts, u)

    def surprise(u, i):
        return s_ui.loc[u, i][0]

    def users_surprised_by(i):
        return s_iu.loc[i].index
        # return safe_loc_index(s_iu, i)

    def items_surprising(u):
        return s_ui.loc[u].index


    @cache
    def items_surprising_both(u, v):
        if u > v:
            return items_surprising_both(v, u)
        else:
            return items_surprising(u).intersection(items_surprising(v))

    @cache
    def correlated_friends(u):
        return filter_index(
            friends(u).intersection(s_u),
            lambda v: len(items_surprising_both(u, v)) > 0.)

    r_m = max(rs) - min(rs)

    @cache
    def sc(u, v):
        if u > v:
            return sc(v, u)
        else:
            return 1 - sum(abs(surprise(u, i) - surprise(v, i)) for i in items_surprising_both(u, v)) / \
                   (len(items_surprising_both(u, v)) * r_m)

    def friends_surprised_by(u, i):
        return correlated_friends(u).intersection(users_surprised_by(i))

    def ss(u, i):
        fseff = friends_surprised_by(u, i)
        return sum(sc(u, v) * surprise(v, i) for v in fseff) / \
               len(fseff)

    def ss_pool(u):
        return correlated_friends(u).rename('v').join(
            s_ui.index.rename(['v', 'item_id']),
            how='inner'
        ).droplevel(0).unique().sort_values()

    @cache
    def friends_rated(u, i):
        return friends(u).intersection(users_rated(i))

    def c(u, i, j):
        return len(filter_index(
            friends_rated(u, i),
            lambda v: rating(v, i) == j))

    @cache
    def c_sum(u, i):
        return len(friends_rated(u, i))

    def p(u, i, j):
        return c(u, i, j) / \
               c_sum(u, i)

    def se(u, i):
        return - sum(xlogx(p(u, i, j)) for j in rs)

    r = len(rs)

    def ds(u, i):
        return r / (c_sum(u, i) + r)

    def us(u, i):
        return (1. - ds(u, i)) * se(u, i)

    # NOTE: CS

    # Caution: Could be None!
    def cs(u, i):
        fri = friends_rated(u, i)
        len_fri = len(fri)
        r_avg = sum(rating(v, i) for v in fri) / len_fri
        return math.sqrt(
            sum(pow(rating(v, i) - r_avg, 2) for v in fri) /
            len_fri)

    return CuriosityScoring(
        ss,
        us,
        cs,
        ss_pool
    )