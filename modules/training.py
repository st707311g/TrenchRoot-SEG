import os
import random

import numpy as np
from skimage import color, exposure
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .__initialize import separated_root_image_dir, separated_trace_image_dir, training_data_dir


def preprocessing(ary) -> np.ndarray:
    ary = exposure.rescale_intensity(ary, out_range=(random.randint(0, 31), random.randint(160, 255)))
    ary = exposure.adjust_gamma(ary, gamma=random.uniform(0.95, 1.05))
    ary = np.array(color.rgb2hsv(ary))
    ary[:, :, 1] = ary[:, :, 1] + random.uniform(-0.2, 0.2)
    ary = color.hsv2rgb(ary)  # * 255.0
    ary = np.clip(ary, 0, 255).astype("uint8")
    return ary


def get_train_generator(seed=1, batch_size=10):
    IN_DIR = training_data_dir
    X_NAME = os.path.basename(separated_root_image_dir)
    Y_NAME = os.path.basename(separated_trace_image_dir)

    data_gen_args = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="reflect",
    )

    image_datagen = ImageDataGenerator(**data_gen_args, preprocessing_function=preprocessing)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow_from_directory(IN_DIR, classes=[X_NAME], seed=seed, batch_size=batch_size, class_mode=None)
    mask_generator = mask_datagen.flow_from_directory(
        IN_DIR, classes=[Y_NAME], seed=seed, batch_size=batch_size, class_mode=None, color_mode="grayscale"
    )

    train_generator = zip(image_generator, mask_generator)

    for ximg, yimg in train_generator:
        ximg = np.array(ximg, dtype=np.float32) / 255
        yimg = (np.array(yimg) >= 128) * 1.0
        yield (ximg, yimg)
