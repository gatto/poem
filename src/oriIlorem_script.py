import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from skimage.color import gray2rgb
from skimage import feature, transform

import json
import gzip
import pickle

from skimage.color import gray2rgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from numpy import linalg as LA
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.datasets import mnist, cifar10, fashion_mnist
import sys

sys.path.append('/home/carlometta/ABELE-master/ilore')
from ilorem import ILOREM
from util import neuclidean

sys.path.append('/home/carlometta/ABELE-master/experiments')
from exputil import get_dataset
from exputil import get_black_box
from exputil import get_autoencoder

sys.path.append('/home/carlometta/ABELE-master/autoencoders')
from adversarial import AdversarialAutoencoderMnist, AdversarialAutoencoderCifar10, AdversarialAutoencoderISIC
from variational import VariationalAutoencoderMnist, VariationalAutoencoderCifar10


import warnings
warnings.filterwarnings('ignore')


def main():
    # Load ISIC Dataset
    print('Loading ISIC 2019 Dataset')
    file = open("/home/carlometta/ABELE-master/Dataset/ListDataResize_Sorted.txt", "r")

    file_lines = file.read()
    filelist = file_lines.split("\n")

    def get_X_orig(filelist, i):
        img = plt.imread(filelist[i])

        return img

    def get_Y_orig(labels_loc):
        Y_df = pd.read_csv(labels_loc)
        Y_orig = np.array(Y_df.iloc[:, 1:9])
        return Y_orig

    labels_loc = '/home/carlometta/ABELE-master/Dataset/ISIC_2019_Training_GroundTruth.csv'

    Y_orig = get_Y_orig(labels_loc)

    train_size = int(len(filelist) * 0.8)
    test_size = int(len(filelist) * 0.2)
    mask = random.sample(range(len(filelist)), test_size)
    list_test = np.array(filelist)
    list_test = list_test[mask]

    mask_train = []

    for i in range(len(filelist)):
        if not i in mask:
            mask_train.append(i)

    list_train = np.array(filelist)[mask_train]
    Y_test = Y_orig[mask]
    Y_train = Y_orig[mask_train]
    NUM_TRAIN_IMAGES = len(list_train)
    NUM_TEST_IMAGES = len(list_test)

    print('Creating Generator for ISIC 2019 Dataset')
    # Create Generator for Isic Dataset
    batch_size = 64
    epochs = 10000
    num_classes = 8
    classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

    def image_generator(filelist, batch_size, mode="train", aug=None):
        while True:
            images = []
            # keep looping until we reach our batch size
            while len(images) < batch_size:
                index = random.randrange(0, len(filelist))
                image = get_X_orig(filelist, index)
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
        rotation_range=180,
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
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=True,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    trainGen = image_generator(list_train, batch_size, mode="train", aug=aug)
    testGen = image_generator(list_test, batch_size, mode="train", aug=None)

    #ILOREM

    random_state = 0
    dataset = 'ISIC'
    black_box = 'DNN'

    ae_name = 'aae'

    path = '/home/carlometta/ABELE-master'
    path_models = path + '/models/'
    path_aemodels = path + '/aemodels/%s/%s/' % (dataset, ae_name)

    black_box_filename = path_models + '%s_%s' % (dataset, black_box)

    use_rgb = True
    class_name = 'class'

    model_filename = black_box_filename + '.json'
    weights_filename = black_box_filename + '_weights.h5'

    '''
    bb = model_from_json(open(model_filename, 'r').read())
    bb.load_weights(weights_filename)
    bb.summary()
    '''

    # load json and create model
    print('Loading model from disk')
    json_file = open('/home/carlometta/ABELE-master/models/ISIC_DNN.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    bb = model_from_json(loaded_model_json)
    # load weights into new model
    bb.load_weights('/home/carlometta/ABELE-master/models/ISIC_DNN_weights.h5')
    print('Model loaded from disk')
    print('Model summary')
    bb.summary()

    import scipy

    def bb_predict(X):
        if X.ndim ==3:
            X = np.expand_dims(X, axis = 0)
            X = X.astype('float32') / 255.
            X = np.transpose(X, (0, 3, 1, 2))

            Z = bb.predict(X)
            return np.argmax(Z,axis=1)

        else:
            X = X.astype('float32') / 255.
            X = np.transpose(X, (0, 3, 1, 2))

            Z = bb.predict(X)
            return np.argmax(Z,axis=1)


    def bb_predict_proba(X):
         if X.ndim ==3:
            X = np.expand_dims(X, axis = 0)
            X = X.astype('float32') / 255.
            X = np.transpose(X, (0, 3, 1, 2))

            Z = bb.predict(X)
            return scipy.special.softmax(Z)

         else:
            X = X.astype('float32') / 255.
            X = np.transpose(X, (0, 3, 1, 2))

            Z = bb.predict(X)
            return scipy.special.softmax(Z)

    def transform(X):
        X = X.astype('float32') / 255.
        return X

    ae = get_autoencoder(next(trainGen), ae_name, dataset, path_aemodels)
    ae.load_model()
    '''
    cont=0
    for i in range(len(filelist)):
        if Y_orig[i,7]>0:
            img = plt.imread(filelist[i])
            if bb_predict(img)==7:
                cont+=1
                print("predicted index: ", i)
        #if i % 100 ==0:
            #print('prediction: ', i)
    print('cont = ', cont)
    '''

    #index_tr = random.randrange(0, len(filelist))
    index_tr = 156
    img = plt.imread(filelist[index_tr])
    plt.imshow(img)
    plt.savefig('/home/carlometta/ABELE-master/aemodels/ISIC/aae/explanation/img_to_explain_%s.png' %index_tr, dpi=150)
    #plt.show()

    class_values = ['0', '1', '2', '3', '4', '5', '6', '7']

    explainer = ILOREM(bb_predict, class_name, class_values, neigh_type='rnd', use_prob=True, size=1000, ocr=0.1,
                       kernel_width=None, kernel=None, autoencoder=ae, use_rgb=use_rgb, valid_thr=0.5,
                       filter_crules=True, random_state=random_state, verbose=True, alpha1=0.5, alpha2=0.5,
                       metric=neuclidean, ngen=10, mutpb=0.2, cxpb=0.5, tournsize=3, halloffame_ratio=0.1,
                       bb_predict_proba=bb_predict_proba)

    latent_dim=256

    exp = explainer.explain_instance(img, num_samples=500, use_weights=True, metric=neuclidean)

    print('e = {\n\tr = %s\n\tc = %s    \n}' % (exp.rstr(), exp.cstr()))
    print(exp.bb_pred, exp.dt_pred, exp.fidelity)
    print(bb_predict(img))
    print(bb_predict_proba(img))
    print(exp.limg)

    cprototypes = exp.get_counterfactual_prototypes(eps=0.01)
    cont=0
    for cpimg in cprototypes:
        bboc = bb_predict(np.array([cpimg]))[0]
        plt.imshow(cpimg)
        plt.title('cf - black box %s' % bboc)
        plt.savefig('/home/carlometta/ABELE-master/aemodels/ISIC/aae/explanation/cprototypes_%s_%s.png' % (index_tr,cont), dpi=150)
        #plt.show()
        cont=cont+1

    prototypes = exp.get_prototypes_respecting_rule(num_prototypes=3)
    cont=0
    for pimg in prototypes:
        bbo = bb_predict(np.array([gray2rgb(pimg)]))[0]
        plt.imshow(pimg)
        plt.title('prototype %s' % bbo)
        plt.savefig('/home/carlometta/ABELE-master/aemodels/ISIC/aae/explanation/prototypes_%s_%s.png' % (index_tr,cont), dpi=150)
        #plt.show()
        cont=cont+1

    img2show, mask = exp.get_image_rule(features=None, samples=10)
    plt.imshow(img2show, cmap='gray')
    bbo = bb_predict(np.array([img2show]))[0]
    plt.title('image to explain - black box %s' % bbo)
    plt.savefig('/home/carlometta/ABELE-master/aemodels/ISIC/aae/explanation/get_image_rule.png', dpi=150)
    #plt.show()


    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, img2show.shape[1], dx)
    yy = np.arange(0.0, img2show.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)

    # Compute edges (to overlay to heatmaps later)
    percentile = 100
    dilation = 3.0
    alpha = 0.8
    xi_greyscale = img2show if len(img2show.shape) == 2 else np.mean(img2show, axis=-1)
    #in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
    in_image_upscaled=xi_greyscale
    edges = feature.canny(in_image_upscaled).astype(float)
    edges[edges < 0.5] = np.nan
    edges[:5, :] = np.nan
    edges[-5:, :] = np.nan
    edges[:, :5] = np.nan
    edges[:, -5:] = np.nan
    overlay = edges

    # abs_max = np.percentile(np.abs(data), percentile)
    # abs_min = abs_max

    # plt.pcolormesh(range(mask.shape[0]), range(mask.shape[1]), mask, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    plt.imshow(mask, extent=extent, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    plt.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    plt.axis('off')
    plt.title('attention area respecting latent rule')
    plt.savefig('/home/carlometta/ABELE-master/aemodels/ISIC/aae/explanation/saliency_%s.png' % index_tr, dpi=200)
    #plt.show()


    #plt.figure(figsize=(12, 4))
    #for i in range(latent_dim):
     #img2show, mask = exp.get_image_rule(features=[i], samples=10)
    #plt.subplot(1, 4, i+1)
    #if use_rgb:
      #plt.imshow(img2show)
    #else:
     #plt.imshow(img2show, cmap='gray')
        #plt.pcolormesh(range(mask.shape[0]), range(mask.shape[1]), mask, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
     #plt.title('varying dim %d' % i)
    #plt.suptitle('attention area respecting latent rule')
    #plt.show()


    #prototypes = exp.get_prototypes_respecting_rule(num_prototypes=5, eps=255*0.25)


    # prototypes, diff_list = exp.get_prototypes_respecting_rule(num_prototypes=5, return_diff=True)
    # for pimg, diff in zip(prototypes, diff_list):
    #     bbo = bb_predict(np.array([gray2rgb(pimg)]))[0]
    #     plt.subplot(1, 2, 1)
    #     if use_rgb:
    #         plt.imshow(pimg)
    #     else:
    #         plt.imshow(pimg, cmap='gray')
    #     plt.title('prototype %s' % bbo)
    #     plt.subplot(1, 2, 2)
    #     plt.title('differences')
    #     if use_rgb:
    #         plt.imshow(pimg)
    #     else:
    #         plt.imshow(pimg, cmap='gray')
    #     plt.pcolormesh(range(diff.shape[0]), range(diff.shape[1]), diff, cmap=plt.cm.BrBG, alpha=1, vmin=0, vmax=255)
    #     plt.show()
    #
    #
    # cprototypes_interp = exp.get_counterfactual_prototypes(eps=0.01, interp=5)
    # for cpimg_interp in cprototypes_interp:
    #     for i, cpimg in enumerate(cpimg_interp):
    #         bboc = bb_predict(np.array([cpimg]))[0]
    #         plt.subplot(1, 5, i+1)
    #         if use_rgb:
    #             plt.imshow(cpimg)
    #         else:
    #             plt.imshow(cpimg, cmap='gray')
    #         plt.title('%s' % bboc)
    #     fo = bb_predict(np.array([cpimg_interp[0]]))[0]
    #     to = bb_predict(np.array([cpimg_interp[-1]]))[0]
    #     plt.suptitle('black box - from %s to %s' % (fo, to))
    #     plt.show()

if __name__ == "__main__":
    main()