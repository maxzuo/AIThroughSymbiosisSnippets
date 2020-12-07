import os
import argparse
import datetime

import numpy as np
import cv2

from sklearn.cluster import KMeans
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt

from dist import new_point

