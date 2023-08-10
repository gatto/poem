# MSc Thesis in Data Science

This project includes a contribution on the ABELE explainability pipeline. TODO: insert citation

# API reference

## Example usage

```py
from oab import Explainer, TestPoint
exp = Explainer(TestPoint.generate_test())
```

```py
import numpy as np
from oab import Explainer

a = np.ndarray(…)
exp = Explainer.from_array(a)
```

### oab.Explainer

```py
oab.Explainer(testpoint: TestPoint, howmany: int = 3, save: bool = False)
```

- `testpoint: oab.TestPoint`, the record to be explained
- `howmany`: how many exemplars to generate
- `save`: used for testing, whether to save explanations to disk

#### attributes

- `.target: oab.TreePoint` the TreePoint identified to be the most similar point to the provided testpoint
- `.counterfactuals: list[oab.ImageExplanation]` the counterfactuals: points that are as similar as possible to the provided testpoint but whose blackbox classification results in a different predicted class than testpoint's predicted class
- `.factuals: list[oab.ImageExplanation]` the factuals (aka prototypes): points that are as different as possible to the provided testpoint but whose blackbox classification results in the same predicted class than testpoint's predicted class
- `.eps_factuals: list[oab.ImageExplanation]` another way to compute factuals

### oab.TestPoint

```py
oab.TestPoint(a: np.ndarray, blackbox: oab.Blackbox, domain: oab.Domain)
```

- `a` the array that represents the image to explain in 3-channel encoding (height, width, 3) e.g. (28, 28, 3)
- `blackbox` an oab.Blackbox that contains the blackbox model that provides a .fit() and a .predict() and a predicted_class for the record
- `domain` is generally provided automatically for the problem domain

#### attributes

- `.latent: oab.Latent` the latent representation of the TestPoint. Can call .latent.a to get the numpy.ndarray for the representation.

### oab.ImageExplanation

```py
oab.ImageExplanation(latent: oab.Latent, blackbox: oab.Blackbox)
```

#### attributes

- `.a: np.ndarray` the real-space representation of the image

# Shell commands

## start coding

```bash
cd repos/tesi
source venv/bin/activate
conda activate tesi
```

# development notes

## list of readmes

- `./README.md`
- `src/README.md`
- `src/data/README.md`

## Links

### Tutorials

#### neural networks

- [intro to keras, ottima ma complessa](https://keras.io/getting_started/intro_to_keras_for_researchers/)

## Packets

tensorflow
notebook
seaborn
scikit-learn
scikit-image
sklearn-json
rich
deap

## config

jupyter notebook --generate-config
in ~/.jupyter/jupyter_notebook_config.py, change:
    c.NotebookApp.custom_display_url = '<http://epimelesi.isti.cnr.it:8888>'

## how are generated exemplars/counterexemplars

Exemplars are generated randomly and then discriminated. One by one are verified if can be exemplars, then taken or rejected as exemplars. This goes until we have enough exemplars or max attempts.
counterexemplars: take the rule to falsify, set the corresponding attribute as falsified (to the decision tree boundary) + a small epsilon.

intorno 100, 5 casi su 100 in cui le cose vanno storte. (ovvero se le regole sono vuote)
le spiegazioni proposte (non genera regola, non genera controregola, fallisce la classificazione del DT)

## times

of an execution of 1:22 minutes:seconds, neighgen_fn(img, num_samples) takes 1:12

## stuff todo

TODO: extract and set the Domain.classes array.

TODO: representation of Latent.a is the actual array if it's shorter than e.g. 10, <np.ndarray> if it's longer

## some development notes

```json
{
    'rstr': <ilore.rule.Rule object at 0x7fb76737feb0>,
    'cstr': '{ { 3 <= 0.89 } --> { class: 2 }, { 1 <= 0.55 } --> { class: 4 }, { 3 <= -0.45 } --> { class: 0 }, { 1 <= -0.62 } --> { class: 4 } }',
    'bb_pred': 6,
    'dt_pred': 6,
    'fidelity': 1.0,
    'limg': array([ 0.7269481 ,  1.16434   , -0.42264208,  2.3466258 ], dtype=float32),
    'dt': DecisionTreeClassifier(max_depth=16, min_samples_leaf=0.001, min_samples_split=0.002)
}
```

- **rstr**
- **cstr**
- Blackbox.predicted_class:int **bb_pred**
- LatentDT.predicted_class:int **dt_pred**
- LatentDT.fidelity:float **fidelity**
- Latent.a:np.ndarray  **limg**
- LatentDT.model:sklearn.tree._classes.DecisionTreeClassifier **dt**

## additional TODO

√ Domain

- classes (is the list of strings of classes in the domain problem)

√ LatentDT

- predicted_class
- model
- fidelity

√ Blackbox
-predicted class

### at end of development

- set s_rules and s_counterrules to repr=False in class LatentDT

## dir of `ae: abele.adversarial.AdversarialAutoencoderMnist = get_autoencoder()` object

- 'alpha',
- 'autoencoder',
- 'build_decoder',
- 'build_discriminator',
- 'build_encoder',
- 'decode',
- 'decoder',
- 'discriminate',
- 'discriminator',
- 'encode',
- 'encoder',
- 'fit',
- 'generate',
- 'generator',
- 'hidden_dim',
- 'img_denormalize',
- 'img_normalize',
- 'init',
- 'input_dim',
- 'latent_dim',
- 'load_model',
- 'name',
- 'path',
- 'sample_images',
- 'save_graph',
- 'save_model',
- 'shape',
- 'store_intermediate',
- 'verbose'
