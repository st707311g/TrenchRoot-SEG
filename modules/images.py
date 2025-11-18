from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

import numpy as np
from skimage import io


class Image2D(object):
    def __init__(
        self,
        image: np.ndarray,
        block_size: int = 256,
        overlap: int = 16,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.image = image
        self.block_size = block_size
        self.overlap = overlap

        assert self.block_size > 0
        assert self.overlap >= 0

        self.logger.debug(f"{self.image.shape=}")
        self.logger.debug(f"{self.block_size=}")
        self.logger.debug(f"{self.padding_size=}")
        self.logger.debug(f"{self.padded_image_shape=}")

    @property
    def actual_block_size(self):
        return self.block_size - self.overlap * 2

    @property
    def padding_size(self):
        ret = []
        for i in range(2):
            s = self.image.shape[i]
            b = self.actual_block_size

            if s % b == 0:
                ret.append((0, 0))
            else:
                ret.append((0, b - s % b))

        if self.image.ndim == 3:
            ret.append((0, 0))
        return tuple(ret)

    @property
    def padded_image_shape(self):
        ret = []
        for i in range(self.image.ndim):
            ret.append(self.image.shape[i] + self.padding_size[i][0] + self.padding_size[i][1])

        return tuple(ret)

    def separate(self) -> List[np.ndarray]:
        img = self.image
        img = np.pad(img, self.padding_size, mode="reflect")
        if img.ndim == 3:
            img = np.pad(img, ((self.overlap, self.overlap), (self.overlap, self.overlap), (0, 0)), mode="reflect")
        else:
            img = np.pad(img, ((self.overlap, self.overlap), (self.overlap, self.overlap)), mode="reflect")

        y_total = self.padded_image_shape[0] // self.actual_block_size
        x_total = self.padded_image_shape[1] // self.actual_block_size
        separated_images = []
        for yi in range(y_total):
            for xi in range(x_total):
                cropped_img = img[
                    yi * self.actual_block_size : (yi + 1) * self.actual_block_size + self.overlap * 2,
                    xi * self.actual_block_size : (xi + 1) * self.actual_block_size + self.overlap * 2,
                ]

                separated_images.append(cropped_img)

        self.logger.debug(f"number of image tiles: {len(separated_images)}")

        return separated_images

    def assemble(self, separated_images: list):
        dst_image = np.zeros(self.padded_image_shape[0:2], dtype=np.uint8)

        y_total = self.padded_image_shape[0] // self.actual_block_size
        x_total = self.padded_image_shape[1] // self.actual_block_size
        i = 0
        for yi in range(y_total):
            for xi in range(x_total):
                dst_image[
                    yi * self.actual_block_size : (yi + 1) * self.actual_block_size,
                    xi * self.actual_block_size : (xi + 1) * self.actual_block_size,
                ] = separated_images[i][self.overlap : -self.overlap, self.overlap : -self.overlap]
                i += 1

        dst_image = dst_image[0 : self.image.shape[0], 0 : self.image.shape[1]]

        return dst_image


class ImageSeparator2D(object):
    def __init__(self, indir: str, outdir: str) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.indir = Path(indir)
        self.outdir = Path(outdir)

    def separate(self, block_size=256):
        assert self.indir.is_dir()
        if self.outdir.is_dir():
            shutil.rmtree(self.outdir)

        self.outdir.mkdir(exist_ok=True, parents=True)

        files = list(sorted(self.indir.glob("*.*")))
        files = [f for f in files if f.is_file()]
        files = [f for f in files if f.suffix in (".png", ".jpg", ".tif", ".tiff")]

        for img_file in files:
            try:
                image = io.imread(img_file)
            except:  # noqa
                continue

            relative_path = img_file.relative_to(self.indir)

            self.logger.info(f"Separating image file: {img_file}")
            images = Image2D(image=image).separate()
            for i, img in enumerate(images):
                dst_dir = self.outdir / relative_path
                dst_file = dst_dir / f"img_{i:02}.png"
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                io.imsave(dst_file, img)

        return self

    def get_training_image_number(self):
        return len(list(self.outdir.glob("**/*.png")))
