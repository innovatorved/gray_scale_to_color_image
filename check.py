import math
import tensorflow as tf
from keras import applications
from keras.models import load_model
import os
import numpy as np
import cv2


MODEL_DIR = os.path.join("/content/drive/MyDrive/MODEL/")

PRETRAINED = "my_model_colorization.h5"

VGG_modelF = applications.vgg16.VGG16(weights="imagenet", include_top=True)
save_path = os.path.join(MODEL_DIR, PRETRAINED)
colorizationModel = load_model(save_path)


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


def predict_colored_image(img):
    avg_ssim = 0
    avg_psnr = 0
    batchX, batchY, original, labimg_oritList = generate_batch(img)
    if batchX.any():
        i = 0
        predY, _ = colorizationModel.predict(np.tile(batchX, [1, 1, 1, 3]))
        predictVGG = VGG_modelF.predict(np.tile(batchX, [1, 1, 1, 3]))
        loss = colorizationModel.evaluate(
            np.tile(batchX, [1, 1, 1, 3]), [batchY, predictVGG], verbose=0
        )
        originalResult = original[i]
        height, width, channels = originalResult.shape
        predY_2 = deprocess(predY[i])
        predY_2 = cv2.resize(predY_2, (width, height))
        labimg_oritList_2 = labimg_oritList[i]
        predResult_2 = reconstruct(deprocess(labimg_oritList_2), predY_2)
        ssim = tf.keras.backend.eval(
            tf.image.ssim(
                tf.convert_to_tensor(originalResult, dtype=tf.float32),
                tf.convert_to_tensor(predResult_2, dtype=tf.float32),
                max_val=255,
            )
        )
        psnr = tf.keras.backend.eval(
            tf.image.psnr(
                tf.convert_to_tensor(originalResult, dtype=tf.float32),
                tf.convert_to_tensor(predResult_2, dtype=tf.float32),
                max_val=255,
            )
        )
        avg_ssim += ssim
        avg_psnr += psnr
        return predResult_2, avg_ssim, avg_psnr


import gradio as gr

title = "Black&White to Color image"
description = "Transforming Black & White Image in to colored image. Upload a black and white image to see it colorized by our deep learning model."

gr.Interface(
    fn=predict_colored_image,
    title=title,
    description=description,
    inputs=[gr.Image(label="Gray Scale Image")],
    outputs=[
        gr.Image(label="Predicted Colored Image"),
        gr.Text(label="SSIM Loss"),
        gr.Text(label="PSNR Loss"),
    ],
).launch(share=True, debug=True)
