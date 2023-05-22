import tensorflow as tf
from tensorflow import keras
from rich import print

# x = tf.random.uniform(shape=(20, 2), minval=0, maxval=10, dtype="int32")
# print(x)

(x_train, labels_train), (x_test, labels_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
miao = [x_train, labels_train, x_test, labels_test]

for element in miao:
    print(f"This is the type: {type(element)}")
    print(f"This is the shape: {element.shape}")
    print(f"This is the dataset: {element}")
