# Import Dependencies
import copy
import os
import random
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageFile
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

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


def adversarial(image, label, model):
    x_tensor = tf.Variable(image, dtype=tf.float32)

    with tf.GradientTape() as t:
        t.watch(x_tensor)
        output = model(x_tensor)
        loss = loss_object(label, output)

    gradients = t.gradient(loss, x_tensor)
    gradients = gradients.numpy()
    gradients = np.sign(gradients)
    return gradients


def FGSM(image, label, model):
    epsilon = 0
    while epsilon == 0:
        epsilon = abs(np.round(np.random.normal(0, 8, 1), 0))
    if epsilon > 16:
        epsilon = 16

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


def generateAdversarialBatch(maskDirectory, listOfMasks, imageDirectory, listOfImages, batchSize, model):
    cleanImagesInBatch = int(batchSize / 2)
    amountOfImages = len(listOfImages)

    X_tr = np.zeros((batchSize, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_tr = np.zeros((batchSize, IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

    while 1:
        batchStartIndex = 0
        batchEndIndex = cleanImagesInBatch

        while batchStartIndex < amountOfImages:
            endIndex = 0
            if batchEndIndex < amountOfImages:
                endIndex = batchEndIndex
            else:
                endIndex = amountOfImages

            # Load in batch of images, create adversarial images for the last half on batch
            i = 0
            for image_name in listOfImages[batchStartIndex:batchEndIndex]:
                img_path = imageDirectory + '/' + image_name
                img = imread(img_path, as_gray=True)[:, :]
                img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
                img = np.expand_dims(img, axis=-1)
                X_tr[i] = img
                i = i + 1

            i = 0
            for mask_name in listOfMasks[batchStartIndex:batchEndIndex]:
                mask_path = maskDirectory + '/' + mask_name
                mask = imread(mask_path, as_gray=True)[:, :]
                mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True),
                                      axis=-1)
                mask = 1.0 * (mask > 0.01)
                Y_tr[i] = mask
                Y_tr[i + cleanImagesInBatch] = mask
                i = i + 1

            a = 0
            for i in range(cleanImagesInBatch, batchSize):
                readyAdvImage = copy.deepcopy(X_tr[a])
                readyAdvImage = np.expand_dims(readyAdvImage, axis=0)
                readyAdvMask = copy.deepcopy(Y_tr[i])
                readyAdvMask = np.expand_dims(readyAdvMask, axis=0)
                advImage = FGSM(readyAdvImage, readyAdvMask, model)
                X_tr[i] = advImage
                a = a + 1

            yield X_tr, Y_tr

            batchStartIndex = batchStartIndex + cleanImagesInBatch
            batchEndIndex = batchEndIndex + cleanImagesInBatch


if __name__ == "__main__":

    IMAGE_PATH = r'path'.replace("\\", "/")  # train image path
    MASK_PATH = r'path'.replace("\\", "/")  # train mask path

    # Get train IDs
    train_ids_img = next(os.walk(IMAGE_PATH))[2]

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids_img), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids_img), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

    sys.stdout.flush()

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    for n, id_ in tqdm(enumerate(train_ids_img), total=len(train_ids_img)):
        img_path = IMAGE_PATH + '/' + id_
        img = imread(img_path, as_gray=True)[:, :]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        img = np.expand_dims(img, axis=-1)
        X_train[n] = img

        mask_path = MASK_PATH + '/' + id_
        mask = imread(mask_path, as_gray=True)[:, :]
        mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask = 1.0 * (mask > 0.01)
        Y_train[n] = mask

    IMAGE_PATH_VAL = r'path'.replace("\\", "/")  # train (validaiton) image path
    MASK_PATH_VAL = r'path'.replace("\\", "/")  # train (validaiton) image path
    val_ids_img = next(os.walk(IMAGE_PATH_VAL))[2]

    # Get and resize val images and masks
    X_val = np.zeros((len(val_ids_img), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_val = np.zeros((len(val_ids_img), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

    sys.stdout.flush()

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    for n, id_ in tqdm(enumerate(val_ids_img), total=len(val_ids_img)):
        img_path = IMAGE_PATH_VAL + '/' + id_
        img = imread(img_path, as_gray=True)[:, :]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        img = np.expand_dims(img, axis=-1)
        X_val[n] = img

        mask_path = MASK_PATH_VAL + '/' + id_
        mask = imread(mask_path, as_gray=True)[:, :]
        mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask = 1.0 * (mask > 0.01)
        Y_val[n] = mask

    val_data = (X_val, Y_val)

    model_filename = 'Adv.h5'  # Base model
    model_path = (r'path'.replace('\\', '/') + model_filename)
    model = load_model(model_path)

    model.trainable = True  # freeze all layers

    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])

    new_model_filename = 'test.h5' # Now model
    new_model_name = 'test'
    new_model_path = (r'path'.replace('\\', '/') + new_model_filename)

    history_logger = tf.keras.callbacks.CSVLogger(new_model_filename, separator=",", append=True)
    model_checkpoint = ModelCheckpoint(new_model_path, monitor='val_loss', verbose=1, save_best_only=True)

    dataGen = generateAdversarialBatch(MASK_PATH, train_ids_img, IMAGE_PATH, train_ids_img, 16, model)
    valDataGen = generateAdversarialBatch(MASK_PATH_VAL, val_ids_img, IMAGE_PATH_VAL, val_ids_img, 16, model)
    batchSize = 16
    stepsEpoch = len(train_ids_img) // (batchSize / 2)
    valStepsEpoch = len(val_ids_img) // (batchSize / 2)
    history = model.fit(dataGen, steps_per_epoch=stepsEpoch, epochs=100, verbose=1, validation_data=valDataGen,
                        validation_steps=valStepsEpoch, callbacks=[model_checkpoint])

    # history = model.fit(X_train, Y_train, batch_size=16, epochs=100, validation_data=val_data, callbacks=[model_checkpoint]) # Used for clean training

    model.save(new_model_path)
    log_path = (r'path'.replace('\\', '/') + new_model_filename)
    graph_path = r'path'.replace('\\', '/')

    # Loss
    fig = plt.figure(figsize=(10, 8))
    plt.title("Learning curve: " + new_model_name)
    plt.grid(True)
    plt.plot(history.history["loss"], label="loss", color="b")
    plt.plot(history.history["val_loss"], label="val_loss", color="g")
    plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r",
             label="best model")
    plt.xlabel("Epochs")
    plt.legend()
    fig.savefig(graph_path + new_model_name + '_loss')
    plt.close()
