
import os

import numpy as np
from skimage import io

from modules import __initialize
from modules.__initialize import (logger, root_image_dir,
                                  separated_root_image_dir,
                                  separated_trace_image_dir, trace_image_dir,
                                  version)
from modules.images import ImageSeparator2D
from modules.training import get_train_generator

if __name__ == '__main__':
    logger.info(f'TrenchRoot-SEG version {version}')
    logger.info(f'Performing data augmentation test.')

    #// separate training image files
    is2d_root = ImageSeparator2D(indir=root_image_dir, outdir=separated_root_image_dir).separate()
    is2d_trace = ImageSeparator2D(indir=trace_image_dir, outdir=separated_trace_image_dir).separate()

    train_generator = get_train_generator(batch_size=36)
    images = train_generator.__next__()[0]

    outdir = 'results/da_test'
    os.makedirs(outdir, exist_ok=True)
    for i, img in enumerate(images):
        fname = os.path.join(outdir, f'img{i:02}.png')
        img = np.array(img*255, dtype=np.uint8)
        io.imsave(fname, img)

    logger.info(f'Result images were stored at: {outdir}')

