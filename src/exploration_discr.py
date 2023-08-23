import random

import matplotlib.pyplot as plt
import numpy as np
import oab
import pandas as pd

(X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = oab.get_data()
my_dom = oab.Domain(dataset="mnist")


probabilities = []
for array in X_test:
    my_point = oab.TestPoint(a=np.asarray(array), domain=my_dom)
    probabilities.append(my_dom.ae.discriminate(my_point))
plt.hist(probabilities)
plt.savefig("./data/histogram_real_test.png")


probabilities2 = []
for _ in range(10000):
    values = [random.gauss(mu=0.0, sigma=1.0) for x in range(4)]
    my_point = oab.ImageExplanation(latent=oab.Latent(np.asarray(values)))
    probabilities2.append(my_dom.ae.discriminate(my_point))
plt.hist(probabilities2)
plt.savefig("./data/histogram_random_gauss.png")

print("For test points:")
print(pd.Series(probabilities).describe())

print("For random points:")
print(pd.Series(probabilities2).describe())
