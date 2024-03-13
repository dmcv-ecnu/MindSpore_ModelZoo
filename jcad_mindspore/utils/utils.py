import smtplib
import yaml
import cv2
import numpy as np

from easydict import EasyDict as ed

def load_config(config_dir):
    return ed(yaml.load(open(config_dir), yaml.FullLoader))