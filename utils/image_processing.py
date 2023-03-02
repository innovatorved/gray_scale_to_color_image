import numpy as np
import cv2
import math


total_batch = 1
data_index = 0
size = 1
batch_size = 1


def read_img(img):
    IMAGE_SIZE = 224
    MAX_SIDE = 1500
    if img is None:
        return False, False, False, False, False
    height, width, channels = img.shape
    if height > MAX_SIDE or width > MAX_SIDE:
        r = min(MAX_SIDE / height, MAX_SIDE / width)
        height = math.floor(r * height)
        width = math.floor(r * width)
        img = cv2.resize(img, (width, height))
    labimg = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
    labimg_ori = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    return (
        True,
        np.reshape(labimg[:, :, 0], (IMAGE_SIZE, IMAGE_SIZE, 1)),
        labimg[:, :, 1:],
        img,
        np.reshape(labimg_ori[:, :, 0], (height, width, 1)),
    )


def generate_batch(img):
    data_index = 0
    batch = []
    labels = []
    labimg_oritList = []
    originalList = []
    ok, greyimg, colorimg, original, labimg_ori = read_img(img)
    if ok:
        batch.append(greyimg)
        labels.append(colorimg)
        originalList.append(original)
        labimg_oritList.append(labimg_ori)
    batch = np.asarray(batch) / 255  # values between 0 and 1
    labels = np.asarray(labels) / 255  # values between 0 and 1
    originalList = np.asarray(originalList)
    labimg_oritList = np.asarray(labimg_oritList) / 255
    return batch, labels, originalList, labimg_oritList


def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)


def reconstruct(batchX, predictedY):
    result = np.concatenate((batchX, predictedY), axis=2)
    result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
    return result
