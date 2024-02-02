# %% [markdown]
# # Timing of oab
# ## Run these once, at the start of the notebook

# %%
import logging
from time import perf_counter

import numpy as np
import pandas as pd
from rich import print

import oab

# %%
# make results list
results_table = []

# %% [markdown]
# ## Personalize the following cell

# %%
# personalize this for running the notebook in different ways
algo_name = "RF"

how_many_images = 3  # how many images each combination of the above?

# %% [markdown]
# ## Run these after each "personalization"
# To get results on multiple datasets, algorithms, classes

# %%
my_dom = oab.Domain("mnist", "RF")


# %%
start = perf_counter()
my_dom.load()
end = perf_counter()
print(end - start)

# %%
print(len(my_dom.explanation_base))
print(type(my_dom.explanation_base[0]))

# %% [markdown]
# ### Timed cells

# %%
datasets = ["mnist", "fashion", "emnist"]

for dataset_name in datasets:
    (X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = oab.get_data(dataset_name)
    my_dom = oab.Domain(dataset_name, algo_name)
    index = {}
    points = {}
    for my_class in range(10):
        index[my_class] = np.where(Y_test == my_class)[0]
    my_dom.load()

    for my_class in [int(x) for x in my_dom.classes]:
        for i, my_index in enumerate(index[my_class]):
            start = perf_counter()
            testpoint = oab.TestPoint(X_test[my_index], my_dom)
            logging.info(f"start explanation of index {my_index}")
            exp = oab.Explainer(testpoint, howmany=5)

            if not exp.factuals:
                exp.factuals = []
            print(f"# factuals made: {len(exp.factuals)}")
            # counterfactuals part
            if not exp.counterfactuals:
                exp.counterfactuals = []
            print(f"how many counterfactuals made: {len(exp.counterfactuals)}")

            end = perf_counter()
            execution_time = end - start
            # append into results_table
            results_table.append(
                {
                    "Dataset": dataset_name,
                    "Algo": algo_name,
                    "Class": my_class,
                    "id": my_index,
                    "Time": execution_time,
                    "factuals / 5": len(exp.factuals),
                    "cfact": len(exp.counterfactuals),
                    "crules": len(exp.target.latentdt.counterrules),
                }
            )

            if i >= how_many_images:
                # how many points to do per class
                break

# %% [markdown]
# ## Run this to show the results table

# %%
results_dataframe = pd.DataFrame.from_records(results_table)
results_dataframe

# %%
