# coding: utf-8
import sys, os
PROJECT_PATH = '/home/ec2-user/ML_Tips/20_DNN/scratch'
sys.path.append(PROJECT_PATH)

import numpy as np
import pickle
from collections import OrderedDict
from common.layers import *

# mypy
from typing import List, Set, Dict, Tuple, Optional, NewType
from mypy_extensions import TypedDict

input_dim = TypedDict('input_dim', {'C': int, 'H': int, 'W': int})
conv_param = TypedDict('conv_param', {'filter_num': int, 'filter_size': int, 'pad': int, 'stride': int})

class DeepConvNet:
    """認識率99%以上の高精度なConvNet
    Note:
        ネットワーク構成は下記の通り
        conv1 - relu1 - conv2 - relu2 - pool - 
        conv3 - relu3 - conv4 - relu4 - pool - 
        conv5 - relu5 - conv6 - relu6 - pool - 
        affine - relu7 - dropout - affine - dropout - softmax
    """
    def __init__(self, input_dim:input_dim = {'C':1, 'H':28, 'W':28},
                conv_param_1:conv_param = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                conv_param_2:conv_param = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                conv_param_3:conv_param = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                conv_param_4:conv_param = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                conv_param_5:conv_param = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                conv_param_6:conv_param = {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                hidden_size:int = 50, output_size:int = 10) -> None:
                # 重みの初期化
                # 各層のニューロンひとつあたりが、前層のニューロンといくつのつながりがあるか（TODO:自動で計算する）
                # それぞれ，[relu1, relu2, relu3, relu4, relu5, relu6, relu7, softmax]に入ってくる前層のニューロン数
                pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
                weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLUを使う場合に推奨される初期値（He）

                self.params: Dict[str, np.ndarray] = {}
                pre_channel_num: int = input_dim['C']
                for idx, conv_param in enumerate([conv_param_1,
                                                  conv_param_2,
                                                  conv_param_3,
                                                  conv_param_4,
                                                  conv_param_5,
                                                  conv_param_6
                                                  ]):
                        self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'],
                                                                                                  pre_channel_num,
                                                                                                  conv_param['filter_size'],
                                                                                                  conv_param['filter_size'])
                        self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
                        pre_channel_num = conv_param['filter_num']


                self.params['W7'] = weight_init_scales[6] * np.random.randn(64*4*4, hidden_size)
                self.params['b7'] = np.zeros(hidden_size)
                self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, output_size)
                self.params['b8'] = np.zeros(output_size)

                # レイヤの生成============
                self.layers = []
                self.layers.append()
