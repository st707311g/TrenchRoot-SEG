import argparse
import logging
import os
import warnings
from pathlib import Path

import coloredlogs
from keras.callbacks import CSVLogger

from modules.config import (
    root_image_dir,
    separated_root_image_dir,
    separated_trace_image_dir,
    trace_image_dir,
    version,
)
from modules.images import ImageSeparator2D
from modules.model import UNet
from modules.training import get_train_generator

logger = logging.getLogger(Path(__file__).name)
coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training TrenchRoot-SEG model")
    parser.add_argument("-b", "--batchsize", type=int, default=16, help="batch size for training")
    parser.add_argument("-e", "--epochs", type=int, default=500, help="epochs for training")
    parser.add_argument("-w", "--weight", type=str, default="", help="pretrained weight file [.hdf5]")
    parser.add_argument("-v", "--version", action="store_true", help="show version information")
    args = parser.parse_args()

    logger.info(f"TrenchRoot-SEG version {version}")

    if args.version:
        exit(0)

    # // separate training image files
    is2d_root = ImageSeparator2D(indir=root_image_dir, outdir=separated_root_image_dir).separate()
    is2d_trace = ImageSeparator2D(indir=trace_image_dir, outdir=separated_trace_image_dir).separate()

    training_image_number = is2d_root.get_training_image_number()
    batch_size = args.batchsize
    epochs = args.epochs
    weight_file = args.weight

    weight_file = weight_file if os.path.isfile(weight_file) else ""

    logger.info(f"training image number: {training_image_number}, batch size: {args.batchsize}, epochs {args.epochs}, weight file: {weight_file}")
    train_generator = get_train_generator(batch_size=batch_size)
    unet = UNet(input_shape=(256, 256, 3), pretrained_weights=weight_file)
    model = unet.model()
    model.summary()

    Path("results").mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger("results/trainlog.csv")

    hist = model.fit(train_generator, steps_per_epoch=training_image_number // batch_size, epochs=epochs, callbacks=[csv_logger])
    model.save_weights("results/TrenchRoot-SEG.hdf5")
