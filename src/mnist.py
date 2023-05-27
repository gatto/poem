user_names = ["Fabio", "Carlo"]
current_user = user_names[0]
# current_user = user_names[1]    # <<<<------------

import json
import logging
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")


try:
    run_options = sys.argv[1]
except IndexError:
    raise Exception(
        """possible runtime arguments are:\n
        understanding, delete-all, train-aae, train-bb, explain <index_image_to_explain>"""
    )


import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf

# from oab import TrainPoint
from ilore.ilorem import ILOREM
from ilore.util import neuclidean
from keras.preprocessing.image import ImageDataGenerator
from oriexputil import get_autoencoder, get_black_box, train_black_box
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


def empty_folder(my_path: str | Path):
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


logger = logging.getLogger()
console = Console()


# Setting general variables
logger.setLevel(logging.INFO)

random_state = 0
dataset = "mnist"
black_box = "RF"
use_rgb = False  # g with mnist dataset


gpus = tf.config.list_physical_devices("GPU")
logging.info(f"Do we even have a gpu? {'no 🤣' if len(gpus)==0 else gpus}")
if len(gpus) > 1:
    logging.info("the code is not optimized for more than 1 gpu")

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


if run_options == "delete-all":
    # Only run if you want this to start over, or you are running this for the first time to create the data folders.
    # g CARE! THIS DELETES ALL FILES IN THE INPUT DIRECTORIES. Run if you want to start over completely
    empty_folder("./data/aemodels/mnist/aae/explanation")
    empty_folder("./data/aemodels/mnist/aae")
    empty_folder("./data/models")
    empty_folder("./data/results/bb")
    exit(0)

# # Build Dataset

# Load X_train, Y_train, X_test, Y_test from mnist keras dataset
console.print("Loading dataset")

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

# TODO: save dataset to file

# # Load Dataset

# TODO: load dataset from csv

# # Data understanding
if run_options == "understanding":
    console.print("Data understanding")
    print(f"X_train[18][18]: {X_train[18][18]}")
    print(f"X_tree[18][18]: {X_tree[18][18]}")  # g they different

table = Table(title="Datasets")
table.add_column("Name", style="cyan", no_wrap=True)
table.add_column("dType", style="magenta")
table.add_column("shape", style="magenta")
datasets = ["X_train", "X_test", "X_tree", "Y_train", "Y_test", "Y_tree"]
for dataset_name in datasets:
    variable = globals()[dataset_name]
    table.add_row(f"{dataset_name}", f"{type(variable)}", f"{variable.shape}")
console.print(table)

# # Training autoencoder
if run_options == "train-aae":
    ae_name = "aae"
    batch_size = 256
    sample_interval = 200

    epochs = 10000  # g time intensive

    path_aemodels = f"data/aemodels/{dataset}/{ae_name}/"

    ae = get_autoencoder(X_train, ae_name, dataset, path_aemodels)

    ae.fit(
        X_train, epochs=epochs, batch_size=batch_size, sample_interval=sample_interval
    )
    ae.save_model()
    ae.sample_images(epochs)

    notify_task(current_user, False, task="autoencoder")

    # # Testing autoencoder with rmse (xxx check with Carlo)
    ae = get_autoencoder(
        X_test, ae_name, dataset, path_aemodels
    )  # g this sets up the autoencoder but does not fit it
    ae.load_model()
    X_train_ae = ae.decode(ae.encode(X_train))
    X_test_ae = ae.decode(ae.encode(X_test))

    table = Table(title=f"Evaluation encoder over {dataset} dataset")
    table.add_column("Series", style="cyan", no_wrap=True)
    table.add_column("mean", style="dark_blue")
    table.add_column("RMSE", style="magenta")
    table.add_column("min", style="green")
    table.add_column("max", style="red")

    table.add_row("train rmse", "", f"{round(rmse(X_train, X_train_ae), 4)}", "", "")
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
if run_options == "train-bb":
    print(f"{black_box} black box training on {dataset} with use_rgb: {use_rgb}")
    print(f"X_train.shape: {X_train.shape}")
    print(f"X_test.shape: {X_test.shape}")

    path = "./"
    path_models = path + "data/models/"
    path_results = path + "data/results/bb/"

    black_box_filename = path_models + "%s_%s" % (dataset, black_box)
    results_filename = path_results + "%s_%s.json" % (dataset, black_box)

    train_black_box(
        X_train, Y_train, dataset, black_box, black_box_filename, use_rgb, random_state
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
    res = {"dataset": dataset, "black_box": black_box, "accuracy": acc, "report": cr}
    results = open(results_filename, "w")
    results.write("%s\n" % json.dumps(res, sort_keys=True, indent=4))
    results.close()
# end black box training

# explain an image
if run_options == "explain":
    NUM_TRAIN_IMAGES = len(X_train)
    NUM_TEST_IMAGES = len(X_test)
    NUM_IMAGES = NUM_TRAIN_IMAGES + NUM_TEST_IMAGES
    print(f"We have {NUM_IMAGES} images")

    batch_size = 64
    epochs = 10000  # was 10k
    class_values = ["%s" % i for i in range(len(np.unique(Y_test)))]
    num_classes = len(class_values)
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

    Y_pred = bb_predict(X_test)
    print(classification_report(Y_test, Y_pred))

    ae = get_autoencoder(
        X_test, ae_name, dataset, path_aemodels
    )  # was next(trainGen) instead of X_train (xxx X_test or X_train?)
    ae.load_model()

    try:
        index_tr = sys.argv[2]
    except IndexError:
        index_tr = 156
    console.print(f"explaining image #{index_tr} from [red]test[/] set")
    img = X_test[index_tr]
    plt.imshow(img)
    plt.savefig(
        "./data/aemodels/mnist/aae/explanation/img_to_explain_%s.png" % index_tr,
        dpi=150,
    )

    class_name = "class"

    # g explainer was size = 1000
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

    exp = explainer.explain_instance(
        img, num_samples=1000, use_weights=True, metric=neuclidean
    )

    console.print("e = {\n\tr = %s\n\tc = %s}" % (exp.rstr(), exp.cstr()))
    console.print(f"exp.bb_pred: {exp.bb_pred}")
    console.print(f"exp.dt_pred: {exp.dt_pred}")
    console.print(f"exp.fidelity: {exp.fidelity}")

    print(f"exp.limg: {exp.limg}")

    tosave = {
        "rstr": exp.rstr(),
        "cstr": exp.cstr(),
        "bb_pred": exp.bb_pred,
        "dt_pred": exp.dt_pred,
        "fidelity": exp.fidelity,
        "limg": exp.limg
    }

    with open(f"./data/aemodels/mnist/aae/explanation/{index_tr}.json", "w") as f:
        f.write("%s\n" % json.dumps(tosave, sort_keys=True, indent=4))

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
        notify_task(task, False, current_user)
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
        notify_task(current_user, False, task)
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
        notify_task(current_user, False, task)
        exit(1)

    # g and wat tis
    task = "math1"
    print(f"Doing [green]{task}[/]")
    dx, dy = 0.05, 0.05
    try:
        xx = np.arange(0.0, img2show.shape[1], dx)
        yy = np.arange(0.0, img2show.shape[0], dy)
        xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
        extent = xmin, xmax, ymin, ymax
        cmap_xi = plt.get_cmap("Greys_r")
        cmap_xi.set_bad(alpha=0)
    except:
        notify_task(current_user, False, task)
        exit(1)

    task = "math2"
    print(f"Doing [green]{task}[/]")
    # Compute edges (to overlay to heatmaps later)
    percentile = 100
    dilation = 3.0
    alpha = 0.8
    try:
        xi_greyscale = (
            img2show if len(img2show.shape) == 2 else np.mean(img2show, axis=-1)
        )
        # in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        in_image_upscaled = xi_greyscale
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges
    except:
        notify_task(current_user, False, task)
        exit(1)

    print("plot")
    # plt.pcolormesh(range(mask.shape[0]), range(mask.shape[1]), mask, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    plt.imshow(mask, extent=extent, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    plt.imshow(overlay, extent=extent, interpolation="none", cmap=cmap_xi, alpha=alpha)
    plt.axis("off")
    plt.title("attention area respecting latent rule")
    plt.savefig(
        "./data/aemodels/mnist/aae/explanation/saliency_%s.png" % index_tr, dpi=200
    )
    # plt.show()

    notify_task(current_user, True, task="explanation")
# end explain an image
