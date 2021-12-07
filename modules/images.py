import os
import shutil
from typing import List

import numpy as np
from skimage import io

from .__initialize import logger


class Image2D(object):
    def __init__(self, img_file: str):
        assert os.path.isfile(img_file)
        self.img_file = img_file
        self.__in_img: np.ndarray = np.array(io.imread(self.img_file))

        self.__out_img = None
        self.__block_size = None

    def separate(self, block_size=(256,256)) -> List[np.ndarray]:
        img = self.__in_img.copy()

        self.__block_size = block_size
        padding_size = [img.shape[i]-(img.shape[i]//block_size[i])*block_size[i] for i in range(2)]
        if len(img.shape)==3:
            img = np.pad(img, [(0, padding_size[0]), (0, padding_size[1]), (0,0)] , mode='reflect')
        else:
            img = np.pad(img, [(0, padding_size[0]), (0, padding_size[1])] , mode='reflect')

        imgs = []
        for yi in range(img.shape[0]//block_size[0]):
            for xi in range(img.shape[1]//block_size[1]):
                crop = img[yi*block_size[0]:(yi+1)*block_size[0], 
                           xi*block_size[1]:(xi+1)*block_size[1]]
                imgs.append(crop)

        return imgs.copy()

    def assemble(self, separated_imgs: list):
        assert self.__in_img is not None
        assert self.__block_size is not None
        img = self.__in_img.copy()
        block_size = self.__block_size

        self.__out_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        i = 0
        for yi in range(self.__out_img.shape[0]//block_size[0]):
            for xi in range(self.__out_img.shape[1]//block_size[1]):
                self.__out_img[yi*block_size[0]:(yi+1)*block_size[0], xi*block_size[1]:(xi+1)*block_size[1]] = separated_imgs[i]
                i += 1

    def in_img(self):
        self.__in_img
        return self.__in_img

    def out_img(self):
        return self.__out_img

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
            images = Image2D(img_file=in_fname).separate(block_size=block_size)
            logger.info(f'Separating image file: {in_fname}')
            for i, img in enumerate(images):
                io.imsave("%s_%02d.tiff" % (out_fname[:-4], i), img)

        return self

    def get_training_image_number(self):
        return len([f for f in os.listdir(self.outdir) if f.endswith('.tiff')])
