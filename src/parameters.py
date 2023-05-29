from pathlib import Path

ae_name = "aae"
random_state = 0
dataset = "mnist"
black_box = "RF"
use_rgb = False  # g with mnist dataset

path = "./"
path_aemodels = path + "data/aemodels/%s/%s/" % (dataset, ae_name)

path_aemodels = Path(path_aemodels)
