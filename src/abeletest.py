from mnist import get_autoencoder, get_data, get_dataset_metadata, run_explain
import keras.backend as K
from pathlib import Path

(X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = get_data()


mtda = get_dataset_metadata()

print(mtda["path_aemodels"])
print(Path(mtda["path_aemodels"]).exists())



ae = get_autoencoder(
    X_test, mtda["ae_name"], mtda["dataset"], mtda["path_aemodels"]
)
ae.load_model()

print(type(ae))
