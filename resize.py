import matplotlib.pyplot as plt
import numpy as np
import platform
import os
import tensorflow as tf

from glob import glob
from pathlib import Path
from tqdm import tqdm

from skimage.io import imread

from warnings import filterwarnings

filterwarnings(action="ignore", category=FutureWarning)

import os

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
except:
    pass

try:
    from tensorflow.python.util import deprecation

    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass


def findfiles(root: Path, regex: str):
    reg = None
    root = root.resolve()
    if platform.system() != "Windows":
        reg = f"{str(root)}/**/{regex}"
    else:
        reg = f"{str(root)}\\**\\{regex}"
    files = [Path(file) for file in glob(reg, recursive=True)]
    if len(files) == 0:
        raise Exception("Bad regex. No files found.")
    files.sort()
    return files


def mkdirp(path: Path):
    """Just a wrapper around os.makedirs for readability"""
    os.makedirs(path, exist_ok=True)


def resize_imgs(images: [Path]) -> np.array:
    n_imgs = len(images)
    # important to specify dtype=np.unit8, or else huge (15GB) file
    resized_imgs = np.empty([n_imgs, 100, 100, 3], dtype=np.uint8)

    # loop over img paths and resize, save resized in np.array
    for i, (img, folder_name) in tqdm(
        enumerate(zip(images, folder_names)), total=n_imgs, desc="Resizing images...."
    ):
        img = imread(img)
        resized = tf.image.resize(img, (100, 100))
        resized_imgs[i, :, :, :] = resized

    return resized_imgs


# get image and containing folder names
script_folder = Path(__file__).absolute().parent.absolute()
images = findfiles(script_folder, "*.jpg")
print(images)
folder_names = [image.parent.name.lower().replace(" ", "_") for image in images]

# make folder for resized image array
resized_folder = script_folder / "PINS_resized"
mkdirp(resized_folder)

# resize
resized_imgs = resize_imgs(images)

# save images and labels
resized_file = resized_folder / "resized_all"
labels_file = resized_folder / "resized_labels"
print(
    f"Writing resized images to {str(resized_file.absolute().resolve())}"
    ". This might take a little bit of time..."
)
np.save(resized_folder / "resized_all", resized_imgs)
print(f"Writing labels for resized images to {str(labels_file.absolute().resolve())}")
np.save(resized_folder / "resized_labels", np.array(folder_names))

