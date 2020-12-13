from ml_script import *
from itertools import product
import json

# test a set of parameter combinations
logs = []
lr_list = [0.001, 0.003, 0.01, 0.03, 0.1]
bs_list = [4, 8, 16]
for lr, bs in product(lr_list, bs_list):
    log = train(lr=lr, batch_size=bs, num_epochs=30)
    logs.append(log)

with open('experiment.json', 'w') as f:
    json.dump(logs, f)

# 10-fold CV
logs = train(lr=0.01, batch_size=4, num_epochs=50, k_fold=10)
with open('10fold_CV.json', 'w') as f:
    json.dump(logs, f)