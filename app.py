import warnings

warnings.filterwarnings("ignore")

from model import download_model_if_not_exists

import sys

isExist = download_model_if_not_exists()
if isExist == False:
    sys.exit(0)

import gradio as gr
from src import predict_colored_image


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
