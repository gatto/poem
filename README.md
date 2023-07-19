# MSc Thesis in Data Science

# Comandi
## start coding
```bash
cd repos/tesi
source venv/bin/activate
conda activate tesi
```

# list of readmes
- `./README.md`
- `src/README.md`
- `src/data/README.md`

# Links
## Tutorials
### neural networks
- [intro to keras, ottima ma complessa](https://keras.io/getting_started/intro_to_keras_for_researchers/)

# Packets
tensorflow
notebook
seaborn
scikit-learn
scikit-image
sklearn-json
rich
deap

# config
jupyter notebook --generate-config
in ~/.jupyter/jupyter_notebook_config.py, change:
    c.NotebookApp.custom_display_url = 'http://epimelesi.isti.cnr.it:8888'

# pickle of explanation
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

# how are generated exemplars/counterexemplars
Exemplars are generated randomly and then discriminated. One by one are verified if can be exemplars, then taken or rejected as exemplars. This goes until we have enough exemplars or max attempts.
counterexemplars: take the rule to falsify, set the corresponding attribute as falsified (to the decision tree boundary) + a small epsilon.


intorno 100, 5 casi su 100 in cui le cose vanno storte. (ovvero se le regole sono vuote)
le spiegazioni proposte (non genera regola, non genera controregola, fallisce la classificazione del DT)


# times
of an execution of 1:22 minutes:seconds, neighgen_fn(img, num_samples) takes 1:12

# stuff todo
TODO: extract and set the Domain.classes array.
TODO: set both predicted_class as integer indexes of the Domain.classes array.

TODO: representation of Latent.a is the actual array if it's shorter than e.g. 10, <np.ndarray> if it's longer


# some development notes
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

## additional
√ Domain
- classes (is the list of strings of classes in the domain problem)

√ LatentDT
- predicted_class
- model
- fidelity

√ Latent
- a

√ Blackbox
-predicted class

TreePoint
- id
- a

Explainer

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
