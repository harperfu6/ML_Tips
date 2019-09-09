# coding: utf-8
import os, sys
sys.path.append('../')
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet