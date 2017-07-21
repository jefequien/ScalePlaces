import os
import sys
sys.path.append(os.path.abspath('../utils'))
from utils import *


def parse_model(prototxt_path):
    models = ["baseline","image", "canny", "image_complex"]
    for model in models:
        prototxt = os.path.basename(prototxt_path)
        if model in prototxt:
            return model
    raise Exception('Model not found')
