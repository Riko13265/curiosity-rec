
input_csv_flixster = {
    'csv_rating_path': 'H://Projects-SideProjects/SKY/flixster/ratings.txt',
    'csv_rating_opts': {
        'names': ['user_id', 'item_id', 'rating'],
        'sep': '\t', 'index_col': False
    },
    'csv_trust_path': 'H://Projects-SideProjects/SKY/flixster/links.txt',
    'csv_trust_opts': {
        'names': ['u', 'v'],
        'sep': '\t', 'index_col': False
    },
    'trust_u_column': 'u'
}

input_kernelmf_01 = {
    'n_epochs': 5,
    'n_factors': 10,
    'verbose': 1,
    'lr': 0.001,
    'reg': 0.02
}

input_train_ratio = 0.6

input_rs = [0.5,
            1.0, 1.5,
            2.0, 2.5,
            3.0, 3.5,
            4.0, 4.5,
            5.0]