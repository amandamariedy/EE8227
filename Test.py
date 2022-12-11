# Import Dependencies
import copy
import os
import sys
import csv
import random
import warnings
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import ImageFile
from skimage.io import imread
from keras import backend as K
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.models import load_model


plt.rc('image', cmap='gray')
print("Imported all the dependencies")

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

loss_object = tf.keras.losses.BinaryCrossentropy()


def dice_coef(y_true, y_pred, smooth=1): # Use to calcuated F1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.one_hot(K.cast(y_true_f, np.uint8), 2)
    y_pred_f = K.one_hot(K.cast(y_pred_f, np.uint8), 2)
    intersection = K.sum(y_true_f[:, 1:] * y_pred_f[:, 1:], axis=[-1])
    union = K.sum(y_true_f[:, 1:], axis=[-1]) + K.sum(y_pred_f[:, 1:], axis=[-1])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


def adversarial(img, label, model):
    #  img = np.expand_dims(img, axis=0) # uncomment for fgsm
    #  label = np.expand_dims(label, axis=0) # uncomment for fgsm
    x_tensor = tf.Variable(img, dtype=tf.float32)

    with tf.GradientTape() as t:
        t.watch(x_tensor)
        output = model(x_tensor)
        loss = loss_object(label, output)
        # print(loss)

    result = output
    gradients = t.gradient(loss, x_tensor)
    gradients = gradients.numpy()
    gradients = np.sign(gradients)
    return gradients


def FGSM(image, label, model, input):
    epsilon = input

    testingImg = copy.deepcopy(image)
    below = testingImg - epsilon
    below = np.clip(below, 0, 255)
    above = testingImg + epsilon
    above = np.clip(above, 0, 255)

    gradients = adversarial(testingImg, label, model)

    testingImg = testingImg + (epsilon * gradients)
    clipBelow = np.maximum(below, testingImg)
    clipAbove = np.minimum(above, clipBelow)
    testingImg = clipAbove

    return testingImg


def PGD(image, label, model, input):
    image = np.expand_dims(image, axis=0)
    label = np.expand_dims(label, axis=0)
    epsilon = input
    numIterations = 5
    step_size = 1
    testingImg = copy.deepcopy(image)
    below = testingImg - epsilon
    below = np.clip(below, 0, 255)
    above = testingImg + epsilon
    above = np.clip(above, 0, 255)

    gen_img = tf.identity(testingImg)
    gen_img = gen_img + tf.random.uniform(gen_img.get_shape().as_list(), minval=-epsilon, maxval=epsilon,
                                          dtype=tf.dtypes.float64)
    starting_img = gen_img.numpy()
    starting_img = np.clip(starting_img, 0, 255)

    for i in range(numIterations):
        gradients = adversarial(starting_img, label, model)
        starting_img = starting_img + (step_size * gradients)
        clipBelow = np.maximum(below, starting_img)
        clipAbove = np.minimum(above, clipBelow)
        starting_img = clipAbove

    starting_img = np.clip(starting_img, 0, 255)
    starting_img = np.squeeze(starting_img, axis=0)
    return starting_img


def getMetric(model_path,epsilon):
    MASK_PATH = r'ADD'.replace("\\", "/")
    IMAGE_PATH = r'ADD'.replace("\\", "/")
    val_ids_img = next(os.walk(MASK_PATH))[2]
    val_ids_img = val_ids_img[:500]
    model = load_model(model_path)

    # Get and resize val images and masks
    X_val = np.zeros((len(val_ids_img), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_val = np.zeros((len(val_ids_img), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
    sys.stdout.flush()
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    for n, id_ in tqdm(enumerate(val_ids_img), total=len(val_ids_img)):
        img_path = IMAGE_PATH + '/' + id_
        img = imread(img_path, as_gray=True)[:, :]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        img = np.expand_dims(img, axis=-1)
        mask_path = MASK_PATH + '/' + id_
        mask = imread(mask_path, as_gray=True)[:, :]
        mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask = 1.0 * (mask > 0.01)
        Y_val[n] = mask
        X_val[n] = img
        # X_val[n] = PGD(img, mask, model, epsilon)  # Uncomment for PGD attack
        # X_val[n] = FGSM(img, mask, model, epsilon)  # Uncomment for FGSM attack
    predictions = model.predict(X_val, verbose=1)
    predictions = (predictions > 0.5).astype(np.uint8)
    metric = tf.keras.metrics.MeanIoU(num_classes=2)
    metric.update_state(predictions, Y_val)
    return metric.result().numpy()


if __name__ == "__main__":

    row1 = ['Clean', 'Clean_Clean_Tuning', 'Clean_Adv_Tuning', 'Adv', 'Adv_Clean_Tuning', 'Adv_Adv_Tuning']
    row2 = [0]
    row3 = [2]
    row4 = [4]
    row5 = [8]
    row6 = [16]
    model_path = (r'G:\Shared drives\F22\Secure ML\EE8227 Project\U-Net\model/'.replace('\\', '/') + 'Clean.h5') # Path for model 1
    row2.append(getMetric(model_path, 0))
    row3.append(getMetric(model_path, 2))
    row4.append(getMetric(model_path, 4))
    row5.append(getMetric(model_path, 8))
    row6.append(getMetric(model_path, 16))
    model_path = (
            r'G:\Shared drives\F22\Secure ML\EE8227 Project\U-Net\model/'.replace('\\', '/') + 'Clean_Clean_Tuning.h5') # Path for model 2
    row2.append(getMetric(model_path, 0))
    row3.append(getMetric(model_path, 2))
    row4.append(getMetric(model_path, 4))
    row5.append(getMetric(model_path, 8))
    row6.append(getMetric(model_path, 16))
    model_path = (r'G:\Shared drives\F22\Secure ML\EE8227 Project\U-Net\model/'.replace('\\', '/') + 'Clean_Adv_Tuning.h5') # Path for model 3
    row2.append(getMetric(model_path, 0))
    row3.append(getMetric(model_path, 2))
    row4.append(getMetric(model_path, 4))
    row5.append(getMetric(model_path, 8))
    row6.append(getMetric(model_path, 16))
    model_path = (r'G:\Shared drives\F22\Secure ML\EE8227 Project\U-Net\model/'.replace('\\', '/') + 'Adv.h5') # Path for model 4
    row2.append(getMetric(model_path, 0))
    row3.append(getMetric(model_path, 2))
    row4.append(getMetric(model_path, 4))
    row5.append(getMetric(model_path, 8))
    row6.append(getMetric(model_path, 16))
    model_path = (r'G:\Shared drives\F22\Secure ML\EE8227 Project\U-Net\model/'.replace('\\', '/') + 'Adv_Clean_Tuning.h5') # Path for model 5
    row2.append(getMetric(model_path, 0))
    row3.append(getMetric(model_path, 2))
    row4.append(getMetric(model_path, 4))
    row5.append(getMetric(model_path, 8))
    row6.append(getMetric(model_path, 16))
    model_path = (r'G:\Shared drives\F22\Secure ML\EE8227 Project\U-Net\model/'.replace('\\', '/') + 'Adv_Adv_Tuning.h5') # Path for model 6
    row2.append(getMetric(model_path, 0))
    row3.append(getMetric(model_path, 2))
    row4.append(getMetric(model_path, 4))
    row5.append(getMetric(model_path, 8))
    row6.append(getMetric(model_path, 16))

    with open(r'Path to CSV'.replace(
            '\\', '/'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(row1)
        writer.writerow(row2)
        writer.writerow(row3)
        writer.writerow(row4)
        writer.writerow(row5)
        writer.writerow(row6)
