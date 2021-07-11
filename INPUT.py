import os

from dotenv import dotenv_values

config = dotenv_values(".env")

input_csv_flixster = {
    'csv_rating_path': config['FLIXSTER_RATINGS_PATH'],
    'csv_rating_opts': {
        'names': ['user_id', 'item_id', 'rating'],
        'sep': '\t', 'index_col': False
    },
    'csv_trust_path': config['FLIXSTER_TRUSTS_PATH'],
    'csv_trust_opts': {
        'names': ['u', 'v'],
        'sep': '\t', 'index_col': False
    },
    'trust_u_column': 'u',
    'rs': [0.5,
           1.0, 1.5,
           2.0, 2.5,
           3.0, 3.5,
           4.0, 4.5,
           5.0]
}

input_kernelmf_01 = {
    'n_epochs': 20,
    'n_factors': 10,
    'verbose': 1,
    'lr': 0.001,
    'reg': 0.02
}

input_train_ratio = 0.6

