import numpy as np
import pandas as pd
from rich import print

dataset = "mnist"
algos = ("DNN", "RF")

for sizeexpl in (500,):
    match dataset:
        case "mnist" | "fashion":
            classes = [x for x in range(10)]
        case "emnist":
            classes = [x for x in range(1, 10)]  # 1, 27
    for algo in algos:
        first_run = False
        results = None
        for my_class in classes:
            loaded = pd.read_csv(f"time-{dataset}-{algo}-{my_class}-{sizeexpl}.csv")
            loaded["crules"] = loaded["crules"].astype(str)
            if not first_run:
                results = pd.concat((results, loaded), ignore_index=True)
            else:
                results = loaded
                results["crules"] = results["crules"].astype(str)
        print(f"[red]{dataset=}, {algo=}")
        print(results.describe())
        hit_percentage = 1 - (results["crules"].value_counts()["error"] / len(results))
        print(f"[red]Hit percentage:[/] {hit_percentage:.4f}")

# metrics are execution time and hit percentage
