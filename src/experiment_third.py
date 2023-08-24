import logging

logging.basicConfig(
    filename="./data/mnist-oab-exp-third-exp.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
import pickle
import oab
import numpy as np
from rich import print
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = oab.get_data()
my_dom = oab.Domain("mnist")
index = {}
points = {}
for my_class in range(10):
    index[my_class] = np.where(Y_test == my_class)[0]

my_dom.load()

how_many_classes_todo = 2

for my_class in range(2, 2 + how_many_classes_todo):
    print(
        f"There are {len(index[my_class])} points with class {my_class} in the test set"
    )
    for my_index in index[my_class]:
        testpoint = oab.TestPoint(X_test[my_index], my_dom)
        logging.info(f"start explanation of index {my_index}")
        exp = oab.Explainer(testpoint, howmany=10)
        if testpoint in exp.target.latentdt.rule:
            try:
                points[my_class].append(exp)
            except KeyError:
                points[my_class] = [exp]
            print("nice")
            if len(points[my_class]) >= 3:
                # how many points to do per class
                break
    with open(f"./data/oab/exp3/{my_class}.pickle", "wb") as f:
        pickle.dump(points[my_class], f, protocol=pickle.HIGHEST_PROTOCOL)
    del points[my_class]
