import warnings
warnings.filterwarnings("ignore")

import math
import tensorflow as tf
from keras import applications
from keras.models import load_model
import os
import numpy as np
import cv2

from utils import generate_batch, deprocess, reconstruct
from model import download_model_if_not_exists


download_model_if_not_exists()
MODEL_DIR = os.path.join("/workspaces/gray_scale_to_color_image/model")

PRETRAINED = "my_model_colorization.h5"

VGG_modelF = applications.vgg16.VGG16(weights="imagenet", include_top=True)
save_path = os.path.join(MODEL_DIR, PRETRAINED)
colorizationModel = load_model(save_path)


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