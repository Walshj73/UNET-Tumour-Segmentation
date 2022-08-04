import cv2
import os
import numpy as np
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_data(PATH, split=0.1):
    images = sorted(glob(os.path.join(PATH, "UNET-Data/Transversal/*")))
    masks = sorted(glob(os.path.join(PATH, "UNET-Data/Transversal-Mask/*")))

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(
            images,
            test_size=valid_size,
            random_state=42
    )
    train_y, valid_y = train_test_split(
            masks,
            test_size=valid_size,
            random_state=42
    )

    train_x, test_x = train_test_split(
            images,
            test_size=test_size,
            random_state=42
    )
    train_y, test_y = train_test_split(
            masks,
            test_size=test_size,
            random_state=42
    )

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image(PATH):
    PATH = PATH.decode()
    x = cv2.imread(PATH, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (128, 128))
    x = x/255.0
    return x


def read_mask(PATH):
    PATH = PATH.decode()
    x = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (128, 128))
    x = x/255.0
    x = np.expand_dims(x, axis=-1)
    return x


def tf_parse(sx, sy):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    sx, sy = tf.numpy_function(_parse, [sx, sy], [tf.float64, tf.float64])
    sx.set_shape([128, 128, 3])
    sy.set_shape([128, 128, 1])
    return sx, sy


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset


if __name__ == "__main__":
    PATH = "../../Dataset/"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(PATH)

    print("Training data: ", len(train_x))
    print("Validation data: ", len(valid_x))
    print("Testing data: ", len(test_x))
    print(" ")

    ds = tf_dataset(test_x, test_y)
    for x, y in ds:
        print(x.shape, y.shape)
        break
