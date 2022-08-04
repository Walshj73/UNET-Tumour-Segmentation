import os
import cv2
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from keras.utils import CustomObjectScope
from Data import load_data, tf_dataset
from Train import iou

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def read_image(PATH):
    x = cv2.imread(PATH, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (128, 128))
    x = x/255.0
    return x


def read_mask(PATH):
    x = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (128, 128))
    x = np.expand_dims(x, axis=-1)
    return x


def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    PATH = "../../Dataset/"
    BATCH = 8

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(PATH)

    test_dataset = tf_dataset(test_x, test_y, batch=BATCH)
    test_steps = len(test_x) // BATCH

    if len(test_x) % BATCH != 0:
        test_steps += 1

    with CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model("../Models/model_transversal.h5")
    model.evaluate(test_dataset, steps=test_steps)

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        y = read_mask(y)
        y_pred = model.predict(np.expand_dims(x, axis=0))
        y_pred = y_pred[0] > 0.5

        # cv2.imwrite(f"../Evaluation Scans/Full/Scans_{i}.png", x)
        # cv2.imwrite(f"../Masks/FULL_{i}.png", y)
        # cv2.imwrite(f"../Results/FULL_{i}.png", y_pred)

        h, w, _ = x.shape

        white_line = np.ones((h, 10, 3)) * 255.0
        all_images = [
                x * 255.0, white_line,
                mask_parse(y), white_line,
                mask_parse(y_pred) * 255.0
        ]
        image = np.concatenate(all_images, axis=1)
        cv2.imwrite(f"../Results/{i}.png", image)
