



def score_of_rank(rank_x):
    rx = rank_x.sort_values(ascending=False).reset_index().reset_index()
    rx['score'] = len(rank_x) - rx['index']
    return rx.set_index('item_id')['score']


def borda_count(rank_x, rank_y, omega):
    score_x = score_of_rank(rank_x)
    score_y = score_of_rank(rank_y)
    score_merge = (1. - omega) * score_x + omega * score_y
    return score_merge.sort_values(ascending=False).reset_index()['item_id']