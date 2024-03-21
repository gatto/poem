import numpy as np
import pandas as pd
from rich import print

datasets = ("mnist", "fashion", "emnist")  # , "emnist"
algos = ("dnn", "rf")

loaded = pd.read_csv("abele_time.csv")
for dataset in datasets:
    match dataset:
        case "mnist" | "fashion":
            classes = [x for x in range(10)]
        case "emnist":
            classes = [x for x in range(1, 27)]
    for algo in algos:
        results = loaded[(loaded["Dataset"] == dataset) & (loaded["BB"] == algo)]
        print(f"[red]{dataset=}, {algo=}")
        print(results.describe())
