# Module: Curiosity
# Description:
# Input: R(u, i), PR(u, i), Relation(u, v)
# Output: SS(u, i), CS(u, i), US(u, i)
from functools import cache
from typing import NamedTuple, Set, Callable

UserItemField = Callable[[int, int], float]

UserUserSet = Callable[[int], Set[int]]


class CuriosityScoring(NamedTuple):
    ss: UserItemField
    cs: UserItemField
    us: UserItemField


def curiosity_scoring(
        rated_items:
        UserUserSet,
        ratings:
        UserItemField,
        p_ratings:
        UserItemField,
        friends:
        UserUserSet,
        r_m: float
) -> CuriosityScoring:

    @cache
    def s(u, i):
        return max(0., ratings(u, i) - p_ratings(u, i))

    @cache
    def surprising_items(u):
        return [i for i in rated_items(u) if s(u, i) > 0.]

    @cache
    def m(u, v):
        return [i for i in surprising_items(u) & surprising_items(v)]

    @cache
    def sc(u, v):
        return 1 - \
               sum(abs(s(u, i) - s(v, i)) for i in m(u, v)) / \
               (len(m(u, v)) * r_m)

    @cache
    def f_s(u, i):
        return [v for v in friends(u) if s(v, i) > 0.]

    def ss(u, i):
        return sum(sc(u, v) * s(v, i) for v in f_s(u, i)) / \
               len(f_s(u, i))

    return CuriosityScoring(
        ss,
        ss,
        ss
    )
