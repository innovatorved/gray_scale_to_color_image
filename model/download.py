import requests
import time
import gdown


url = ""

buffer_size = 1024 * 1024  # 1 MB
progress_update_frequency = 1  # update progress every 1 second


def download_from_url():
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("Content-Length", 0))

        with open("model/my_model_colorization.h5", "wb") as f:
            bytes_downloaded = 0
            start_time = time.time()
            for chunk in response.iter_content(chunk_size=buffer_size):
                f.write(chunk)
                bytes_downloaded += len(chunk)
                time_elapsed = time.time() - start_time
                download_speed = bytes_downloaded / time_elapsed

                # Print the download progress status
                if time_elapsed >= progress_update_frequency:
                    progress = min(100, bytes_downloaded / total_size * 100)
                    print(
                        f"Downloaded {bytes_downloaded} / {total_size} bytes ({progress:.2f}%), "
                        f"speed: {download_speed / 1024 / 1024:.2f} MB/s"
                    )
                    start_time = time.time()
        return True
    except:
        return False


def download_from_drive():
    url = "https://drive.google.com/uc?id=1-2qaNwFjPyzVe3ggfECB4dPhn8rhBCqd"
    try:
        output = "model/my_model_colorization.h5"
        gdown.download(url, output, quiet=False)
        return True
    except:
        print("Error Occured in Downloading model from Gdrive")
        return False
