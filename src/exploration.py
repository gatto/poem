import logging

import matplotlib.pyplot as plt
import oab

logging.basicConfig(
    filename="./data/mnist-oab.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

(X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = oab.get_data()

my_dom = oab.Domain(dataset="mnist")

# this gives problems: oab.TestPoint(X_test[200], domain=my_dom)
points = X_test[:10]

for i, array in enumerate(points):
    logging.info("Point #%s creation", i)
    test_point = oab.TestPoint(array, my_dom)
    logging.info("Point #%s explanation", i)
    exp = oab.Explainer(test_point)
