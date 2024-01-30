import argparse
import os
from glob import glob
from typing import Final

import numpy as np
from skimage import io

from modules import __initialize  # noqa
from modules.__initialize import logger, version
from modules.images import Image2D
from modules.model import UNet

description_text: Final[str] = f"TrenchRoot-SEG (version {version}): A deep learning-based phenotypic analysis tool for trench profile images."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description_text)
    parser.add_argument("-i", "--indir", type=str, default="", help="import a target directory")
    parser.add_argument("-v", "--version", action="store_true", help="show version information")
    args = parser.parse_args()

    logger.info(f"TrenchRoot-SEG version {version}")

    if args.indir == "":
        logger.error("Indicate input directory.")
        exit(1)

    if args.version:
        exit(0)

    unet = UNet(pretrained_weights="TrenchRoot-SEG.hdf5")

    files = sorted(glob(os.path.join(args.indir, "**/*"), recursive=True))
    files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

    for f in files:
        if os.path.splitext(f)[0].endswith("_predicted"):
            continue
        if os.path.isfile(os.path.splitext(f)[0] + "_predicted.png"):
            continue

        try:
            image = io.imread(f)
        except:  # noqa
            logger.error(f"could not read the image: {f}")
            continue

        img = Image2D(image=image)
        separated_imgs = np.array(img.separate()) / 255.0
        predicted = unet.model().predict(separated_imgs, batch_size=1, verbose=1)
        predicted = (predicted[..., 0] * 255).astype(np.uint8)
        assembled_img = img.assemble(separated_images=list(predicted))
        io.imsave(os.path.splitext(f)[0] + "_predicted.png", assembled_img)
