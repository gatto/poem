# MSc Thesis in Data Science

# Comandi
## start coding
```bash
cd repos/tesi
source venv/bin/activate
```

## start remotely
gatto (it's in ~/.local/bin/)

# list of readmes
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
    'limg': array([ 0.7269481 ,  1.16434   , -0.42264208,  2.3466258 ], dtype=float32)
}
```