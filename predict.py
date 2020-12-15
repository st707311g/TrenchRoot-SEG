from model import UNet
import numpy as np
from skimage import io
import argparse
import os, sys, glob
from glob import glob
from model import UNet

from typing import Final
version: Final[str] = '1.0'

description_text: Final[str] = f'TrenchRoot-SEG (version {version}): A deep learning-based phenotypic analysis tool for trench profile images.'

class Image(object):
    def __init__(self):
        self.sep_imgs = None
        self.__in_img = None
        self.__out_img = None
        self.__block_size = None
        self.overlap_size = 0

    def load(self, file: str):
        self.__in_file = file
        self.__in_img = io.imread(self.__in_file)
        self.__range = (0, self.__in_img.shape[0], 0, self.__in_img.shape[1])

    def shape(self):
        return self.__in_img.shape

    def separate(self, block_size=(256,256)):
        assert self.__in_img is not None

        self.__block_size = block_size

        img = self.__in_img.copy()
        padding_size = [img.shape[i]-(img.shape[i]//block_size[i])*block_size[i] for i in range(2)]
        img = np.pad(img, [(0, padding_size[0]), (0, padding_size[1]), (0,0)] , mode='reflect')

        imgs = []
        for yi in range(img.shape[0]//block_size[0]):
            for xi in range(img.shape[1]//block_size[1]):
                crop = img[yi*block_size[0]:(yi+1)*block_size[0], 
                           xi*block_size[1]:(xi+1)*block_size[1]]
                imgs.append(crop)

        return imgs.copy()

    def assemble(self, separated_imgs: list):
        self.__out_img = np.zeros((self.__in_img.shape[0], self.__in_img.shape[1]), dtype=np.uint8)

        block_size = self.__block_size

        i = 0
        for yi in range(self.__out_img.shape[0]//block_size[0]):
            for xi in range(self.__out_img.shape[1]//block_size[1]):
                self.__out_img[yi*block_size[0]:(yi+1)*block_size[0], xi*block_size[1]:(xi+1)*block_size[1]] = separated_imgs[i]
                i += 1

    def in_img(self):
        return self.__in_img

    def out_img(self):
        return self.__out_img

if __name__ == '__main__':
    if sys.version_info.major!=3 or sys.version_info.minor<8:
        raise Exception('Use Python version 3.8 or later.')

    parser = argparse.ArgumentParser(description=description_text)
    parser.add_argument('-i', '--indir', type=str, help='import a target directory')
    parser.add_argument('-v', '--version', action='store_true', help='show version information')
    args = parser.parse_args()

    if args.indir is None:
        parser.print_help()
        sys.exit(1)

    unet = UNet(pretrained_weights='TrenchRoot-SEG.hdf5')

    files = sorted(glob(os.path.join(args.indir, '**/*'), recursive=True))
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    for f in files:
        if os.path.splitext(f)[0].endswith('_predicted'):
            continue
        if os.path.isfile(os.path.splitext(f)[0]+'_predicted.png'):
            continue

        img = Image()
        img.load(f)
        separated_imgs = np.asarray(img.separate()) / 255.
        predicted = unet.model().predict(separated_imgs, batch_size=1, verbose=1)
        predicted = (predicted[...,0]*255).astype(np.uint8)
        img.assemble(separated_imgs=list(predicted))

        assembled_img = img.out_img()
        io.imsave(os.path.splitext(f)[0]+'_predicted.png', assembled_img)
