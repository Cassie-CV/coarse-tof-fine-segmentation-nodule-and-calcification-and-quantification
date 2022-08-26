from .deeplabv3 import DeepLabV3,DeepLabV3Plus
from . import encoders
from . import utils

from .__version__ import __version__

import warnings
warnings.warn('segmentation_models_pytorch does not suppose timm_efficientnet_encoders')