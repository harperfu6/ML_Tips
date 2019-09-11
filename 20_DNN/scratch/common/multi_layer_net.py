# coding: utf-8
import os, sys
sys.path.append('../')
sys.path.append('../../')

import numpy as np
from collections import OrderedDict
from common.layers import *
# from common.gradient import numerical_gradient


class MultiLayerNet:
    """
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
    """
    def __init__(self, input_size, hidden_size_list, output_size,
                activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        # 最後の層
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b' + str(idx)])
        self.last_layer = SoftmaxWithLoss()


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
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1]) # ReLUを使う場合
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1]) #sigmoidを使う場合

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x


    def loss(self, x, t):
        """損失関数
        Args:
            x (ndarray): 入力データ
            t (ndarray): 教師ラベル

        Returns:
            float: 損失関数の値
        """ 
        y = self.predict(x)

        # 全ての隠れ層の重みのL2ノルムを求める
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        # 最後の損失関数にweight decayを加算する
        return self.last_layer.forward(y, t) + weight_decay

    
    def gradient(self, x, t):
        """勾配を求める（バックプロパゲーション）
        Args:
            x (ndarray): 入力データ
            t (ndarray): 教師ラベル

        Returns:
            dict: 各層の勾配を持ったディクショナリ変数
                grads['W1'], grads['W2'], ...は各層の重み
                grads['b1'], grads['b2'], ...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1 # 損失関数Lについて（dL/dL=1）であるため
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads


    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1) # 教師ラベルがonehotの場合はindexを取得

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy