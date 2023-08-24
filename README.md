# offline-abele: a fast library for explainability in images

This project includes a contribution on the ABELE explainability pipeline. TODO: insert citation

## Getting started

### Prerequisites

[Install Tensorflow](https://www.tensorflow.org/install/pip?hl=it) but using version 2.12. The following works:

```bash
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Then, install everything in [requirements.txt](https://github.com/gatto/ds-thesis/blob/main/requirements.txt) as well.

### Installation

I'll present installation instructions for the dataset *mnist*. We assume we work with the default of 10000 instances in the tree_set dataset.

1. Clone locally this git repository.

2. Then, in the root, run:

```bash
cd src
python mnist.py delete-all
python mnist.py train-aae
python mnist.py train-bb
python mnist.py explain 10000

python oab.py mnist delete-all
python oab.py mnist test-train 10000
```

Be advised that

- `python mnist.py train-aae` executes in around 30 minutes;
- `python mnist.py explain 10000` executes in around 9 days with a GPU and takes about 500 MB on disk. It's recommended to batch execute this by substituting the line with this bash script:

```bash
for ((i=0; i<100; i++))
do
 ~/miniconda3/condabin/conda run -n <conda-environment-name> /<path-to-python>/python mnist.py explain 100
done
```

- `python oab.py mnist test-train 10000` takes a few hours.

## Usage

```py
from rich import print
import oab

(X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = oab.get_data()

my_domain = oab.Domain(dataset="mnist")
array = X_test[100]

exp = oab.Explainer.from_array(a=array, domain=my_domain)
print(exp)
```

To visualize the prototypes and counterexemplars:

```py
exp.factuals.draw()
exp.counterfactuals.draw()
```

#### oab.Explainer

```py
oab.Explainer(testpoint: TestPoint, howmany: int = 3, save: bool = False)
```

- `testpoint: oab.TestPoint`, the record to be explained
- `howmany`: how many exemplars to generate
- `save`: used for testing, whether to save explanations to disk

##### attributes

- `.target: oab.TreePoint` the TreePoint identified to be the most similar point to the provided testpoint
- `.counterfactuals: list[oab.ImageExplanation]` the counterfactuals: points that are as similar as possible to the provided testpoint but whose blackbox classification results in a different predicted class than testpoint's predicted class
- `.factuals: list[oab.ImageExplanation]` the factuals (aka prototypes): points that are as different as possible to the provided testpoint but whose blackbox classification results in the same predicted class than testpoint's predicted class
- `.eps_factuals: list[oab.ImageExplanation]` another way to compute factuals

#### oab.TestPoint

```py
oab.TestPoint(a: np.ndarray, blackbox: oab.Blackbox, domain: oab.Domain)
```

- `a` the array that represents the image to explain in 3-channel encoding (height, width, 3) e.g. (28, 28, 3)
- `blackbox` an oab.Blackbox that contains the blackbox model that provides a .fit() and a .predict() and a predicted_class for the record
- `domain` is generally provided automatically for the problem domain

##### attributes

- `.latent: oab.Latent` the latent representation of the TestPoint. Can call .latent.a to get the numpy.ndarray for the representation.

#### oab.ImageExplanation

```py
oab.ImageExplanation(latent: oab.Latent, blackbox: oab.Blackbox)
```

##### attributes

- `.a: np.ndarray` the real-space representation of the image
