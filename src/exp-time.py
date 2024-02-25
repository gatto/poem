import logging
from time import perf_counter

import numpy as np
import pandas as pd
from rich import print
from rich.progress import track

import oab

results_table = []

dataset_name = "mnist"
algo_name = "RF"
my_class = 0

how_many_images = 50  # how many images each combination of the above?

(X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = oab.get_data(dataset_name)
my_dom = oab.Domain(dataset_name, algo_name)
index = {}
points = {}
for my_class in range(10):
    index[my_class] = np.where(Y_test == my_class)[0]

for i, my_index in enumerate(track(index[my_class])):
    start = perf_counter()
    testpoint = oab.TestPoint(X_test[my_index], my_dom)
    logging.info(f"start explanation of index {my_index}")
    exp = oab.Explainer(testpoint, howmany=5)

    end = perf_counter()
    execution_time = end - start

    # if we failed to find a target
    if exp.target:
        crules = len(exp.target.latentdt.counterrules)
    else:
        crules = "error"

    # append into results_table
    results_table.append(
        {
            "Dataset": dataset_name,
            "Algo": algo_name,
            "Class": my_class,
            "id": my_index,
            "Time": round(execution_time, 2),
            "factuals / 5": len(exp.factuals),
            "cfact": len(exp.counterfactuals),
            "crules": crules,
        }
    )

    if i >= how_many_images - 1:
        # how many points to do per class
        break

results_dataframe = pd.DataFrame.from_records(results_table)
print(results_dataframe)

results_dataframe.to_csv(f"time-{dataset_name}-{algo_name}-{my_class}.csv", index=False)
