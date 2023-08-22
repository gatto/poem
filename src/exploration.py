import oab
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename="mnist-oab.log", encoding="utf-8", level=logging.DEBUG)


(X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = oab.get_data()

my_dom = oab.Domain(dataset="mnist")

# this gives problems: oab.TestPoint(X_test[200], domain=my_dom)
points = X_test[:2]

for i, array in enumerate(points):
    logging.info(f"Point #{i} creation")
    test_point = oab.TestPoint(array, my_dom)
    logging.info(f"Point #{i} explanation")
    exp = oab.Explainer(test_point)
