# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from collections import OrderedDict
from typing import List, Set, Dict, Tuple, Optional

from common.layers import *
# from common.gradient import numerical_gradient


class MultiLayerNetExtend:
    """拡張版の全結合による多層ニューラルネットワーク
    Note: 以下の機能をもつ
          - Weight Decay（損失関数に重みの正則化項をつける）
          - Dropout（一部の層を飛ばすことでその重みを学習しない）
          - Batch Normalization（各層で学習後，正規分布に従いデータを分散させる）

    Attributes:
        input_size (int): 入力サイズ（MNISTの場合は784）
        hidden_size_list (list): 隠れ層のニューロンの数のリスト(e.g. [100, 100, 100])
        output_size (int): 出力サイズ（MNISTの場合は10）
        activation (str): 'relu' or 'sigmoid'
        weight_init_std (str): 重みの標準偏差を指定（e.g. 0.01）
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
            「Heの初期値」は「Xavierの初期値」と比較しルート２倍の標準偏差をもつ初期値であるため，分布が偏りがちなreluに適している
        weight_decay_lambda (float): Weight Decay（L２ノルム）の強さ
        use_dropout (boolean): Dropoutを使用するかどうか
        dropout_ration (float): Dropoutの割合
        use_batchNorm (boolean): Batch Normalizationを使用するかどうか
    """

    def __init__(self, input_size, hidden_size_list, output_size,
                activation='relu', weight_init_std='relu', weight_decay_lambda=0,
                use_dropout=False, dropout_ration=0.5, use_batchNorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchNorm = use_batchNorm
        self.params = {}

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成
        # Affine -> BatchNormalization -> Activation -> Dropout
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

            # Batch Normalizationを使う場合
            # 下記パラメータは，正規化した後にさらにシフトするためのパラメータ
            # 学習によって適した値に設定される
            if self.use_batchNorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1]) # データに乗算されるパラメータ
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1]) # データに加算されるパラメータ
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])

            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)

        # ソフトマックスに渡す前
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

        # 最後のレイヤ（softmax）
        self.last_layer = SoftmaxWithLoss()

    # 重みの初期化
    def __init_weight(self, weight_init_std):
        """重みの初期値設定
        Args:
            weight_init_std (str): 重みの標準偏差を指定（e.g. 0.01）
                'relu'または'he'を指定した場合は「Heの初期値」を設定
                'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
                「Heの初期値」は「Xavierの初期値」と比較しルート２倍の標準偏差をもつ初期値であるため，分布が偏りがちなreluに適している
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            # activationに入ってくるデータのサンプル数を分母とする
            if str(weight_init_std).lower in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x: np.ndarray, t: np.ndarray, train_flg: False=bool):
        """損失関数
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        # 各層の重みパラメータのweight decayを算出して，総和をとる
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def gradient(self, x: np.ndarray, t: np.ndarray):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batchNorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta
        
        return grads


    def accuracy(self, X: np.ndarray, T: np.ndarray) -> float:
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1: T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy