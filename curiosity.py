import functools
import math
from functools import cache
from typing import NamedTuple, Set, Callable, List

import pandas

IdIdInt = Callable[[int, int], int]

IdIdFloat = Callable[[int, int], float]

IdToIdSet = Callable[[int], Set[int]]

IdToIndex = Callable[[int], pandas.Index]


class CuriosityScoring(NamedTuple):
    ss: IdIdFloat
    us: IdIdFloat
    cs: IdIdFloat
    pool: IdToIdSet


def filter_index(index, value_key, predicate):
    frame = index.to_frame()[value_key]
    return frame.loc[frame.apply(predicate)].index


def xlogx(x):
    if x == 0.:
        return 0.
    else:
        return x * math.log(x)


def curiosity_scoring(
        rated_items:
        IdToIndex,
        rated_users:
        IdToIndex,
        rating:
        IdIdInt,
        p_rating:
        IdIdFloat,
        friends:
        IdToIndex,
        rs:
        List[int]
) -> CuriosityScoring:
    # NOTE: SS

    # Caution: Could throw if rating does not exist!
    @cache
    def s(u, i):
        return max(0., rating(u, i) - p_rating(u, i))

    @cache
    def surprising_items(u):
        return filter_index(
            rated_items(u), 'item_id',
            lambda iid: s(u, iid) > 0.)

    # NOTE: Reverse index
    @cache
    def surprising_users(i):
        return filter_index(
            rated_users(i), 'user_id',
            lambda uid: s(uid, i) > 0.)

    @cache
    def m(u, v):
        if u > v:
            return m(v, u)
        else:
            return surprising_items(u).intersection(
                surprising_items(v))

    @cache
    def surprising_friends(u):
        return filter_index(
            friends(u), 'v',
            lambda v: len(m(u, v)) > 0.)

    r_m = max(rs) - min(rs)

    @cache
    def sc(u, v):
        if u > v:
            return sc(v, u)
        else:
            muv = m(u, v)
            return 1 - sum(abs(s(u, i) - s(v, i)) for i in muv) / \
                   (len(muv) * r_m)

    def f_s_eff(u, i):
        return surprising_friends(u).intersection(
            surprising_users(i))

    def ss(u, i):
        fseff = f_s_eff(u, i)
        return sum(sc(u, v) * s(v, i) for v in fseff) / \
               len(fseff)

    def ss_pool(u):
        return {i for v in surprising_friends(u) for i in surprising_items(v)}

    # NOTE: No need to evaluate us_pool & cs_pool since ss_pool is stricter.

    # NOTE: US

    @cache
    def friends_rated_i(u, i):
        return friends(u).intersection(rated_users(i))

    def c(u, i, j):
        return len(filter_index(
            friends_rated_i(u, i), 0,
            lambda v: rating(v, i) == j))

    @cache
    def c_sum(u, i):
        return len(friends_rated_i(u, i))

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
        fri = friends_rated_i(u, i)
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
