import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Tuple

import napari
import numpy as np
from skimage import io

from .__initialize import logger


@dataclass(frozen=True)
class Image2D(object):
    image: np.array
    logger: logging.Logger = field(init=False)
    block_size: Tuple[int] = (256,256)

    def __post_init__(self):
        assert self.block_size[0] > 0 and self.block_size[0] > 0
        object.__setattr__(self, 'logger', logging.getLogger(self.__class__.__name__))
        self.logger.debug(f'a class constructed: {self.__class__.__name__}')
        self.logger.debug(f'{self.image.shape=}')
        self.logger.debug(f'{self.block_size=}')
        self.logger.debug(f'{self.padding_size=}')
        self.logger.debug(f'{self.padded_image_shape=}')

    @property
    def padding_size(self):
        return tuple([self.image.shape[i]-(self.image.shape[i]//self.block_size[i])*self.block_size[i] for i in range(2)])

    @property
    def padded_image_shape(self):
        return_value = (self.image.shape[0]+self.padding_size[0], self.image.shape[1]+self.padding_size[1])
        if len(self.image.shape)==3:
            return return_value+(self.image.shape[2],)
        else:
            return return_value

    def separate(self) -> List[np.ndarray]:
        img = self.image

        pad_width = ((0, self.padding_size[0]), (0, self.padding_size[1]))
        if len(img.shape)==3:
            pad_width = pad_width+((0,0),)
        img = np.pad(img, pad_width , mode='reflect')

        separated_images = []
        for yi in range(img.shape[0]//self.block_size[0]):
            for xi in range(img.shape[1]//self.block_size[1]):
                crop = img[yi*self.block_size[0]:(yi+1)*self.block_size[0], 
                           xi*self.block_size[1]:(xi+1)*self.block_size[1]]
                separated_images.append(crop)

        self.logger.debug(f'number of image tiles: {len(separated_images)}')

        return separated_images.copy()

    def assemble(self, separated_images: list):
        output_image = np.zeros(self.padded_image_shape[0:2], dtype=np.uint8)

        i = 0
        for yi in range(output_image.shape[0]//self.block_size[0]):
            for xi in range(output_image.shape[1]//self.block_size[1]):
                output_image[yi*self.block_size[0]:(yi+1)*self.block_size[0], xi*self.block_size[1]:(xi+1)*self.block_size[1]] = separated_images[i]
                i += 1

        output_image = output_image[0:self.image.shape[0], 0:self.image.shape[1]]

        return output_image

class ImageSeparator2D(object):
    def __init__(self, indir: str, outdir: str) -> None:
        super().__init__()
        self.indir = indir
        self.outdir = outdir

    def separate(self, block_size=(256,256)):
        assert os.path.isdir(self.indir)
        if os.path.isdir(self.outdir):
            shutil.rmtree(self.outdir)
        os.makedirs(self.outdir)

        f_list = sorted(os.listdir(self.indir))
        f_list = [i for i in f_list if i.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))]
        in_list = [os.path.join(self.indir, f) for f in f_list]

        out_list = [os.path.join(self.outdir, f) for f in f_list]
        
        for in_fname, out_fname in zip(in_list, out_list):
            try:
                image = io.imread(in_fname)
            except:
                continue

            images = Image2D(image=image).separate()
            logger.info(f'Separating image file: {in_fname}')
            for i, img in enumerate(images):
                io.imsave("%s_%02d.tiff" % (out_fname[:-4], i), img)

        return self

    def get_training_image_number(self):
        return len([f for f in os.listdir(self.outdir) if f.endswith('.tiff')])
