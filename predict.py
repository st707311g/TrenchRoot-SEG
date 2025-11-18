from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Final

import coloredlogs
import numpy as np
from skimage import io

from modules.config import version
from modules.images import Image2D
from modules.model import UNet

logger = logging.getLogger(Path(__file__).name)
coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

description_text: Final[str] = f"TrenchRoot-SEG {version}."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description_text)
    parser.add_argument("-i", "--indir", type=Path, help="indicate a target directory")
    parser.add_argument("-v", "--version", action="store_true", help="show version information")
    args = parser.parse_args()

    if args.version:
        print(version)
        exit(0)

    logger.info(f"TrenchRoot-SEG version {version}")

    if args.indir is None:
        logger.info("Indicate valid path by -i, --indir")
        exit(1)

    indir = Path(args.indir)
    if not indir.is_dir():
        logger.info(f"Indicated path is not a directory: {indir}")
        exit(1)

    unet = UNet(pretrained_weights="TrenchRoot-SEG.hdf5")

    for f in sorted(indir.glob("**/*")):
        if not f.name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        if f.stem.endswith("_predicted"):
            continue

        if (f.parent / (f.stem + "_predicted.png")).is_file():
            continue

        logger.info(f)

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
