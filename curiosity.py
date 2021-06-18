# Module: Curiosity
# Description:
# Input: R(u, i), PR(u, i), Relation(u, v)
# Output: SS(u, i), CS(u, i), US(u, i)
import math
from functools import cache
from typing import NamedTuple, Set, Callable, List

IdIdInt = Callable[[int, int], int]

IdIdFloat = Callable[[int, int], float]

IdToIdSet = Callable[[int], Set[int]]


class CuriosityScoring(NamedTuple):
    ss: IdIdFloat
    us: IdIdFloat
    cs: IdIdFloat


def curiosity_scoring(
        rated_items:
        IdToIdSet,
        ratings:
        IdIdInt,
        p_ratings:
        IdIdFloat,
        friends:
        IdToIdSet,
        rs:
        List[int]
) -> CuriosityScoring:

    # NOTE: SS

    @cache
    def s(u, i):
        return max(0., ratings(u, i) - p_ratings(u, i))

    @cache
    def surprising_items(u):
        return [i for i in rated_items(u) if s(u, i) > 0.]

    def m(u, v):
        if u > v:
            return m(v, u)
        else:
            return [i for i in surprising_items(u) & surprising_items(v)]

    r_m = max(rs) - min(rs)

    @cache
    def sc(u, v):
        if u > v:
            return sc(v, u)
        else:
            muv = m(u, v)
            return 1 - \
                   sum(abs(s(u, i) - s(v, i)) for i in muv) / \
                   (len(muv) * r_m)

    def f_s(u, i):
        return [v for v in friends(u) if s(v, i) > 0.]

    def ss(u, i):
        f_s_ = f_s(u, i)
        return sum(sc(u, v) * s(v, i) for v in f_s_) / \
               len(f_s_)

    # NOTE: US

    def c(u, i, j):
        return len(list(v for v in friends(u) if ratings(v, i) == j))

    @cache
    def friends_rated_i(u, i):
        return [v for v in friends(u) if ratings(v, i) is not None]

    @cache
    def c_sum(u, i):
        return len(list(v for v in friends_rated_i(u, i)))

    def p(u, i, j):
        return c(u, i, j) / c_sum(u, i)

    def se(u, i):
        return - sum(p(u, i, j) * math.log(p(u, i, j)) for j in rs)

    r = max(rs)

    def ds(u, i):
        return r / sum(c(u, i, j) + r for j in rs)

    def us(u, i):
        return (1. - ds(u, i)) * se(u, i)

    # NOTE: CS

    def cs(u, i):
        fri = friends_rated_i(u)
        len_fri = len(fri)
        r_avg = sum(ratings(v, i) for v in fri) / len_fri
        return math.sqrt(
            sum(pow(ratings(v, i) - r_avg, 2) for v in fri) /
            len_fri)


    return CuriosityScoring(
        ss,
        us,
        cs
    )
