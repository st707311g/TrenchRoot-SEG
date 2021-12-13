import logging
import os
import sys
from typing import Final

if sys.version_info.major!=3 or sys.version_info.minor<8:
    raise Exception('Use Python version 3.8 or later.')

logger = logging.getLogger('TrenchRoot-SEG')
logger.setLevel(logging.INFO)

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

try:
    import coloredlogs
    coloredlogs.install(level=logging.INFO)
except:
    pass

import warnings

warnings.filterwarnings('ignore') 

training_data_dir: Final[str] = 'data/for_train'
root_image_dir: Final[str] = os.path.join(training_data_dir, 'root_images')
separated_root_image_dir: Final[str] = os.path.join(training_data_dir, 'root_images_separated')
trace_image_dir: Final[str] = os.path.join(training_data_dir, 'trace_images')
separated_trace_image_dir: Final[str] = os.path.join(training_data_dir, 'trace_images_separated')

version: Final[str] = '1.2'

for d in ['../results', '../data/for_train/root_images', '../data/for_train/trace_images']:
    os.makedirs(os.path.join(os.path.dirname(__file__), d), exist_ok=True)
