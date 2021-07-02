from curiosity import curiosity_scoring

rated_items = {
    1: {1, 2, 3},
    2: {2, 3, 4},
    3: {3, 4, 5}
}

ratings = {
    (1, 1): 3, (1, 2): 2, (1, 3): 3,
               (2, 2): 4, (2, 3): 3, (2, 4): 2,
                          (3, 3): 5, (3, 4): 2, (3, 5): 3
}

p_ratings = {
    (1, 1): 3, (1, 2): 1, (1, 3): 1,
               (2, 2): 3, (2, 3): 1, (2, 4): 1.5,
                          (3, 3): 5, (3, 4): 4, (3, 5): 3
}

friends = {
    1: {2, 3},
    2: {3, 1},
    3: {1, 2}
}

rs = [1, 2, 3, 4, 5]

scorings = curiosity_scoring(
    lambda u: rated_items[u],
    lambda u, i: ratings[(u, i)],
    lambda u, i: p_ratings[(u, i)],
    lambda u: friends[u],
    rs
)

scorings.ss(1, 4)
