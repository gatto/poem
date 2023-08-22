user_names = ["Fabio", "Carlo"]
current_user = user_names[0]
# current_user = user_names[1]    # <<<<------------

import gc
import json
import logging
import pickle
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    try:
        run_options = sys.argv[1]
    except IndexError:
        raise Exception(
            """possible runtime arguments are:\n
            understanding, delete-all, train-aae, train-bb, explain <index_image_to_explain>"""
        )

Path("./data").mkdir(exist_ok=True)


import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
from abele.exputil import get_autoencoder, get_black_box, train_black_box
from ilore.ilorem import ILOREM
from ilore.util import neuclidean
from keras.preprocessing.image import ImageDataGenerator
from parameters import *
from rich import print
from rich.console import Console
from rich.progress import track
from rich.table import Table
from skimage import feature, transform
from skimage.color import gray2rgb
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.datasets import mnist


def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def empty_folder(my_path: str | Path) -> None:
    """
    Creates dir if it doesn't exist. If exists, all files are removed.
    """
    if isinstance(my_path, str):
        my_path = Path(my_path)

    assert isinstance(my_path, Path)

    if not my_path.is_relative_to("./data/"):
        raise ValueError("you just tried something very bad")

    if my_path.exists():
        for p in my_path.iterdir():
            if not p.is_dir():
                logging.info(f"deleted file {p}.")
                p.unlink()
            else:
                logging.info(f"we leave dir {p} alone.")

    my_path.mkdir(parents=True, exist_ok=True)


def notify_task(current_user: str, good: bool, task: str) -> None:
    message = f"✅ Done with {task}." if good else f"❌ Bad {task}."
    if current_user == "Fabio":
        chat_id = "29375109"
        url = f"https://api.telegram.org/bot{tg_api_key.key}/sendMessage?chat_id={chat_id}&text={message}"
        requests.get(url).json()  # this sends the message
    if current_user == "Carlo":
        pass


# # Build Dataset
def get_data(dataset: str = "mnist") -> tuple:
    # Load X_train, Y_train, X_test, Y_test from mnist keras dataset
    print("Loading dataset")

    # 2 different alternatives:
    """
    # carlo:
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data(path="mnist.npz")
    X_train = np.expand_dims(X_train, 3)
    X_test = np.expand_dims(X_test, 3)
    """
    # for grayscale:
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data(path="mnist.npz")
    X_train = np.stack([gray2rgb(x) for x in X_train.reshape((-1, 28, 28))], 0)
    X_test = np.stack([gray2rgb(x) for x in X_test.reshape((-1, 28, 28))], 0)

    # Extract X_tree, Y_tree with random (stable) sampling from X_train, Y_train (todo possible even better to gaussian sample it)
    random.seed("gattonemiao")
    indexes = random.sample(
        range(X_train.shape[0]), X_train.shape[0] // 6
    )  # g get a list of 1/6 indexes of the len of X_train

    indexing_condition = []
    for x in track(range(X_train.shape[0]), description="Sampling X_tree, Y_tree"):
        if x in indexes:
            indexing_condition.append(True)
        else:
            indexing_condition.append(False)
    assert len(indexing_condition) == X_train.shape[0]

    logging.info(
        f"We have False number of train records and True number of tree records: {Counter(indexing_condition)}"
    )

    indexing_condition = np.array(indexing_condition)

    X_tree = X_train[indexing_condition]
    Y_tree = Y_train[indexing_condition]

    X_train = X_train[~indexing_condition]
    Y_train = Y_train[~indexing_condition]

    return (X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree)


def get_dataset_metadata() -> dict:
    results = dict()
    results["ae_name"] = "aae"
    results["dataset"] = "mnist"

    results[
        "path_aemodels"
    ] = f"./data/aemodels/{results['dataset']}/{results['ae_name']}/"

    return results


def run_explain(index_tr: int, X: np.ndarray, Y: np.ndarray) -> dict:
    """
    This should, at some point, return what I need
    i.e. the ILORE decision tree
    possibly together with the `tosave` dict
    """
    logging.info(f"Start run_explain of {index_tr}")

    class_values = ["%s" % i for i in range(len(np.unique(Y)))]
    print(f"Classes are: {class_values}")

    """
    def image_generator(filelist, batch_size, mode="train", aug=None):
        while True:
            images = []
            # keep looping until we reach our batch size
            while len(images) < batch_size:
                index = random.randrange(0, len(filelist))
                image = filelist[index]
                # trainNoise = np.random.normal(loc=0, scale=50, size=image.shape)
                # trainXNoisy = np.clip(image + trainNoise, 0, 255)
                # image = trainXNoisy.astype(int)
                # if we are evaluating we should now break from our
                # loop to ensure we don't continue to fill up the
                # batch from samples at the beginning of the file
                if mode == "eval":
                    break

                # update our corresponding batches lists
                images.append(image)

            if aug is not None:
                images = next(aug.flow(np.array(images), batch_size=batch_size))

            yield np.array(images)

    aug = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=10,  #g was 180 for circle-like images like skin things
        # randomly shift images horizontally
        width_shift_range=0.2,
        # randomly shift images vertically
        height_shift_range=0.2,
        # set range for random shear
        shear_range=0.3,
        # set range for random zoom
        zoom_range=0.2,
        # set range for random channel shifts
        channel_shift_range=20,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=False,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    trainGen = image_generator(X_train, batch_size, mode="train", aug=aug)
    testGen = image_generator(X_test, batch_size, mode="train", aug=None)
    print(f"X_test.shape: {X_test.shape}")
    """

    # ILOREM
    ae_name = "aae"

    path = "./"
    path_models = path + "data/models/"
    path_aemodels = path + "data/aemodels/%s/%s/" % (dataset, ae_name)
    black_box_filename = path_models + "%s_%s" % (dataset, black_box)

    bb_predict, bb_predict_proba = get_black_box(
        black_box, black_box_filename, use_rgb
    )  # g this loads bb to disk and returns 2 functs

    # Y_pred = bb_predict(X)
    # print(classification_report(Y, Y_pred))

    ae = get_autoencoder(
        X, ae_name, dataset, path_aemodels
    )  # was next(trainGen) instead of X_train (xxx X_test or X_train?)
    ae.load_model()

    print(f"explaining image #{index_tr} from supplied dataset")
    img = X[index_tr]
    plt.imshow(img)
    plt.savefig(
        "./data/aemodels/mnist/aae/explanation/img_to_explain_%s.png" % index_tr,
        dpi=150,
    )

    class_name = "class"

    explainer = ILOREM(
        bb_predict,
        class_name,
        class_values,
        neigh_type="rnd",
        use_prob=True,
        size=1000,
        ocr=0.1,
        kernel_width=None,
        kernel=None,
        autoencoder=ae,
        use_rgb=use_rgb,
        valid_thr=0.5,
        filter_crules=True,
        random_state=random_state,
        verbose=False,
        alpha1=0.5,
        alpha2=0.5,
        metric=neuclidean,
        ngen=10,
        mutpb=0.2,
        cxpb=0.5,
        tournsize=3,
        halloffame_ratio=0.1,
        bb_predict_proba=bb_predict_proba,
    )

    logging.info("done creating ILOREM")

    exp = explainer.explain_instance(
        img, num_samples=1000, use_weights=True, metric=neuclidean
    )

    logging.info("done .explain_instance")

    print("e = {\n\tr = %s\n\tc = %s}" % (exp.rstr(), exp.cstr()))
    print(f"exp.bb_pred: {exp.bb_pred}")
    print(f"exp.dt_pred: {exp.dt_pred}")
    print(f"exp.fidelity: {exp.fidelity}")

    print(f"exp.limg: {exp.limg}")

    tosave = {
        "rstr": exp.rstr(),
        "cstr": exp.cstr(),
        "bb_pred": exp.bb_pred,
        "dt_pred": exp.dt_pred,
        "fidelity": exp.fidelity,
        "limg": exp.limg,
        "dt": exp.dt,
        "neigh_bounding_box": np.array([np.min(exp.Z, axis=0), np.max(exp.Z, axis=0)]),
    }

    with open(f"./data/aemodels/mnist/aae/explanation/{index_tr}.pickle", "wb") as f:
        pickle.dump(tosave, f, protocol=pickle.HIGHEST_PROTOCOL)

    # this is temporary like the return following
    notify_task(current_user, good=True, task=f"explanation of {index_tr}")

    return tosave

    # xxx continue checking from here
    task = "get_counterfactual_prototypes"
    print(f"Doing [green]{task}[/]")
    cont = 0
    try:
        cprototypes = exp.get_counterfactual_prototypes(eps=0.01)
        for cpimg in cprototypes:
            bboc = bb_predict(np.array([cpimg]))[0]
            plt.imshow(cpimg)
            plt.title("cf - black box %s" % bboc)
            plt.savefig(
                "./data/aemodels/mnist/aae/explanation/cprototypes_%s_%s.png"
                % (index_tr, cont),
                dpi=150,
            )
            # plt.show()
            cont = cont + 1
    except:
        notify_task(current_user, good=False, task=task)
        exit(1)
    print(f"I made #{cont} {task}.")

    task = "get_prototypes_respecting_rule"
    print(f"Doing [green]{task}[/]")
    cont = 0
    try:
        prototypes = exp.get_prototypes_respecting_rule(num_prototypes=3)
        for pimg in prototypes:
            bbo = bb_predict(np.array([pimg]))[0]
            if use_rgb:
                plt.imshow(pimg)
            else:
                plt.imshow(pimg.astype("uint8"), cmap="gray")
            plt.title("prototype %s" % bbo)
            plt.savefig(
                "./data/aemodels/mnist/aae/explanation/prototypes_%s_%s.png"
                % (index_tr, cont),
                dpi=150,
            )
            # plt.show()
            cont = cont + 1
    except:
        notify_task(current_user, good=False, task=task)
        exit(1)
    print(f"I made #{cont} {task}.")

    # g wat is this
    task = "get_image_rule"
    print(f"Doing [green]{task}[/]")
    try:
        img2show, mask = exp.get_image_rule(features=None, samples=10)
        plt.imshow(img2show, cmap="gray")
        bbo = bb_predict(np.array([img2show]))[0]
        plt.title("image to explain - black box %s" % bbo)
        plt.savefig("./data/aemodels/mnist/aae/explanation/get_image_rule.png", dpi=150)
    except:
        notify_task(current_user, good=False, task=task)
        exit(1)

    notify_task(current_user, good=True, task="explanation")


# TODO: save dataset to file

# # Load Dataset

# TODO: load dataset from csv

# user setup
if current_user == "Carlo":
    # !pip install deap

    from google.colab import drive

    drive.mount("/content/gdrive")
    if "/content/gdrive/My Drive/Colab Notebooks/ABELE_prostate/fabio/" not in sys.path:
        sys.path.append(
            "/content/gdrive/My Drive/Colab Notebooks/ABELE_prostate/fabio/"
        )
if current_user == "Fabio":
    import tg_api_key
logging.info(f"✅ Setup for {current_user} done.")


if __name__ == "__main__":
    console = Console()

    gpus = tf.config.list_physical_devices("GPU")
    logging.info(f"Do we even have a gpu? {'no 🤣' if len(gpus)==0 else gpus}")
    if len(gpus) > 1:
        logging.warning("the code is not optimized for more than 1 gpu")

    if run_options == "delete-all":
        # Only run if you want this to start over, or you are running this for the first time to create the data folders.
        # g CARE! THIS DELETES ALL FILES IN THE INPUT DIRECTORIES. Run if you want to start over completely
        empty_folder("./data/aemodels/mnist/aae/explanation")
        empty_folder("./data/aemodels/mnist/aae")
        empty_folder("./data/models")
        empty_folder("./data/results/bb")
        empty_folder("./data/oab")
        empty_folder("./data")
        exit(0)

    # # Data understanding
    if run_options == "understanding":
        (X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = get_data()

        print("Data understanding")
        # print(f"X_train[18][18]: {X_train[18][18]}")
        # print(f"X_tree[18][18]: {X_tree[18][18]}")  # g they different

        table = Table(title="Datasets")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("dType", style="magenta")
        table.add_column("shape", style="magenta")
        datasets = ["X_train", "X_test", "X_tree", "Y_train", "Y_test", "Y_tree"]
        for dataset_name in datasets:
            variable = globals()[dataset_name]
            table.add_row(f"{dataset_name}", f"{type(variable)}", f"{variable.shape}")
        console.print(table)

    # Training autoencoder
    elif run_options == "train-aae":
        (X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = get_data()

        ae_name = get_dataset_metadata()["ae_name"]
        dataset = get_dataset_metadata()["dataset"]
        batch_size = 256
        sample_interval = 200

        epochs = 10000  # g time intensive

        path_aemodels = get_dataset_metadata()["path_aemodels"]

        ae = get_autoencoder(X_train, ae_name, dataset, path_aemodels)

        ae.fit(
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            sample_interval=sample_interval,
        )
        ae.save_model()
        ae.sample_images(epochs)

        notify_task(current_user, good=True, task=run_options)

        # Testing autoencoder with rmse (xxx check with Carlo)
        # g the following sets up the autoencoder but does not fit it
        ae = get_autoencoder(X_test, ae_name, dataset, path_aemodels)
        ae.load_model()
        X_train_ae = ae.decode(ae.encode(X_train))
        X_test_ae = ae.decode(ae.encode(X_test))

        table = Table(title=f"Evaluation encoder over {dataset} dataset")
        table.add_column("Series", style="cyan", no_wrap=True)
        table.add_column("mean", style="dark_blue")
        table.add_column("RMSE", style="magenta")
        table.add_column("min", style="green")
        table.add_column("max", style="red")

        table.add_row(
            "train rmse", "", f"{round(rmse(X_train, X_train_ae), 4)}", "", ""
        )
        table.add_section()
        table.add_row("test rmse", "", f"{round(rmse(X_test, X_test_ae), 4)}", "", "")
        table.add_row(
            "X_test",
            f"{round(np.mean(X_test), 4)}",
            "",
            f"{np.min(X_test)}",
            f"{np.max(X_test)}",
        )
        table.add_row(
            "X_test_ae",
            f"{round(np.mean(X_test_ae), 4)}",
            "",
            f"{np.min(X_test_ae)}",
            f"{np.max(X_test_ae)}",
        )
        table.add_row(
            "X_test - X_test_ae",
            f"{round(np.mean(X_test) - np.mean(X_test_ae), 4)}",
            "",
            f"{round(np.min(X_test) - np.min(X_test_ae), 4)}",
            f"{round(np.max(X_test) - np.max(X_test_ae), 4)}",
        )

        console.print(table)
    # end train autoencoder

    # Black box training (xxx this was with fashion dataset, does it work with mnist?)
    elif run_options == "train-bb":
        (X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = get_data()

        print(f"{black_box} black box training on {dataset} with use_rgb: {use_rgb}")
        print(f"X_train.shape: {X_train.shape}")
        print(f"X_test.shape: {X_test.shape}")

        path = "./"
        path_models = path + "data/models/"
        path_results = path + "data/results/bb/"

        black_box_filename = path_models + "%s_%s" % (dataset, black_box)
        results_filename = path_results + "%s_%s.json" % (dataset, black_box)

        train_black_box(
            X_train,
            Y_train,
            dataset,
            black_box,
            black_box_filename,
            use_rgb,
            967,
        )  # g this fits and saves bb to disk
        bb_predict, bb_predict_proba = get_black_box(
            black_box, black_box_filename, use_rgb
        )  # g this loads bb to disk and returns 2 functs

        Y_pred = bb_predict(X_test)

        acc = accuracy_score(Y_test, Y_pred)
        cr = classification_report(Y_test, Y_pred)
        print("Accuracy: %.4f" % acc)
        print("Classification Report")
        print(cr)
        cr = classification_report(Y_test, Y_pred, output_dict=True)
        res = {
            "dataset": dataset,
            "black_box": black_box,
            "accuracy": acc,
            "report": cr,
        }
        results = open(results_filename, "w")
        results.write("%s\n" % json.dumps(res, sort_keys=True, indent=4))
        results.close()
    # end black box training

    # explain images
    elif run_options == "explain":
        try:
            how_many = int(sys.argv[2])
        except IndexError:
            raise Exception(
                """possible runtime arguments are:
                <how many images to explain>
                example: python mnist.py explain 100
                will explain 100 images not already explained"""
            )

        my_counter = 0
        explanation_path = Path(get_dataset_metadata()["path_aemodels"]) / "explanation"

        # check what I've already done
        hey = {int(path.stem) for path in explanation_path.glob("*.pickle")}
        max_i = max(hey)

        (X_train, Y_train), (X_test, Y_test), (X_tree, Y_tree) = get_data()

        # do some more
        for i in range(max_i + 1, max_i + 1 + how_many):
            _ = run_explain(i, X_tree, Y_tree)
            my_counter += 1
            if my_counter % 10 == 0:
                gc.collect()

        print(
            f"Explained instances from {max_i+1} to {max_i+how_many} amounting to {my_counter} instances."
        )
        # end explain images
