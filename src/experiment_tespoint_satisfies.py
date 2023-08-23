import logging

logging.basicConfig(
    filename="./data/mnist-oab-exp-testpointsatisfies.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
import oab
from rich import print

(X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = oab.get_data()
my_dom = oab.Domain(dataset="mnist")

howmany_satisfy = 0
my_points = X_test

for array in my_points:
    my_point = oab.TestPoint(a=array, domain=my_dom)
    my_tree_point = oab.knn(my_point)
    if my_point in my_tree_point.latentdt.rule:
        howmany_satisfy += 1

print(
    f"------- out of {len(my_points)}, only {howmany_satisfy} satisfy the respective rule --------"
)
