# coding: utf-8
import numpy as np
from typing import List, Set, Dict, Tuple, Optional


class SGD:
    """Stochastic Gradient Descent
    Note: 「サンプルをランダムに１つ選ぶ」というSGDの処理はここでは実装しない
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params: Dict, grads: Dict) -> None:
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    """Momentum SGD
    Note: Momentum=「運動量」．お椀を転がるボールのイメージ，滑らかに収束していくのでSGDより早く収束しやすい
    """
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val) # 速度vは0で初期化
        
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

        
class AdaGrad:
    """AdaGrad(Adaptive Gradient)
    Note: 学習が進むに連れて学習係数を小さくする．更新が大きいパラメータは次回は更新を小さくするので落ち着いた動きをする
    """
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val) # 重みに相関のある係数hは0で初期化
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + 1e-7)) # 0割防止


class RMSprop:
    """RMSprop
    Note: AdaGradは過去の重みを２乗和として全て記録するため，徐々に更新度合いは小さくなる．
          これを改善したもので，過去の全ての勾配を均一に加算するのではなく，新しい勾配の情報が大きく反映させる．
          専門的には，「指数移動平均」を使い，指数関数的に過去の勾配スケールを小さくする．
    """
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val) # 重みに相関のある係数hは0で初期化
            

        for key in params.keys():
            # ここがkey
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            #
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key] + 1e-7)) # 0割防止


class Adam:
    """Adam (Adaptive moment estimation)
    Note: MomentumとAdaGradの融合．
          ハイパーパラメータの「バイアス補正（偏り補正）」を行うことも特徴．
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key] + 1e-7)) # 0割防止