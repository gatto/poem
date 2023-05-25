# files copied pasted

`autoencoders`
- `oriadversarial` no changes
- `oriautoencoder` no changes

`experiments`
- `oriexputil`
    line 188 def get_autoencoder **was verbose=True**
    line 65, 72, 79, 151 X = np.array([rgb2gray(x) for x in X]) if not use_rgb else X **commented out**


`ilore`
- `decision_tree`
    line 27 `dt_search = GridSearchCV(dt, param_grid=param_list, scoring=scoring, cv=cv, n_jobs=-1)` removed `iid=False`
- `explanation`
    line 71 timg = rgb2gray(img2show) if not self.use_rgb else img2show **commented out**
        timg = img2show **inserted**
    line 92 # pimg = rgb2gray(pimg) if not self.use_rgb else pimg **commented out**
    line 95 diff = np.mean(diff, 2) if self.use_rgb else diff **commented out**
- `ilorem` no changes
- `ineighgen` no changes
- `rule` no changes
- `util` no changes

