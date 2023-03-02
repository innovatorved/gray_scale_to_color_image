import os
from .download import download_from_drive


def check_for_file():
    if os.path.exists("model/my_model_colorization.h5"):
        return True
    else:
        return False


def download_model_if_not_exists():
    if check_for_file():
        print("yes")
        return True
    elif download_from_drive():
        print("y33")
        return True
    else:
        print("y3")
        return False
