# Development README

## Shell commands

### start coding

```bash
cd repos/tesi
source venv/bin/activate
conda activate tesi
```

## List of readmes

- `./README.md`
- `src/README.md`
- `src/data/README.md`

## Links

### Tutorials

#### Neural networks

- [intro to keras, ottima ma complessa](https://keras.io/getting_started/intro_to_keras_for_researchers/)

## Packages

tensorflow
notebook
seaborn
scikit-learn
scikit-image
sklearn-json
rich
deap

## Config

jupyter notebook --generate-config
in ~/.jupyter/jupyter_notebook_config.py, change:
    c.NotebookApp.custom_display_url = '<http://epimelesi.isti.cnr.it:8888>'

## How original Abele works

### Exemplars/counterexemplars generation

- Exemplars are generated randomly and then discriminated. One by one are verified if can be exemplars, then taken or rejected as exemplars. This goes until we have enough exemplars or max attempts.
- counterexemplars: take the rule to falsify, set the corresponding attribute as falsified (to the decision tree boundary) + a small epsilon.

### Execution times of Abele

- One explanation: 1:22 minutes:seconds, of which `neighgen_fn(img, num_samples)` takes 1:12

Abele: 600 records explained in exactly 10 hours

## How offline-abele works

### Structure of `tosave`: data saved in mnist.py to .pickle for usage by oab.py

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

### At end of development

- set s_rules and s_counterrules to repr=False in class LatentDT

## Files taken and files modified from original Abele

- `/autoencoders`
  - `adversarial` no changes
  - `autoencoder` no changes
- `/experiments`
  - `exputil` line 188 def get_autoencoder **was verbose=True**
- `/ilore`
  - `decision_tree` line 27 `dt_search = GridSearchCV(dt, param_grid=param_list, scoring=scoring, cv=cv, n_jobs=-1)` removed `iid=False`
  - `explanation`
  - `ilorem` no changes
  - `ineighgen` no changes
  - `rule` no changes
  - `util` no changes
