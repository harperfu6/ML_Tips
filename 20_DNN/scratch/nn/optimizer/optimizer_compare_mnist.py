# coding: utf-8
import os, sys
sys.path.append('../')
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    return x_train, t_train, x_test, x_test

def main():
    # MNISTデータの読み込み
    x_train, t_train, x_test, t_test = get_data()

    # 実験の設定
    train_size = x_train.shape[0]
    batch_size = 128
    max_iterations = 2000

    optimizers = {}
    optimizers['SGD'] = SGD()
    optimizers['Momentum'] = Momentum()
    optimizers['AdaGrad'] = AdaGrad()
    optimizers['Adam'] = Adam()
    optimizers['RMSprop'] = RMSprop()

    netowrks = {}
    train_loss = {}
    for key in optimizers.keys():
        netowrks[key] = MultiLayerNet(
            input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10
        )
        train_loss[key] = []

    # 訓練の開始
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for key in optimizers.keys():
            grads = netowrks[key].gradient(x_batch, t_batch)    
            optimizers[key].update(netowrks[key].params, grads) # 現在のパラメータとその勾配を渡す

            loss = netowrks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        # terminal check
        if i % 100 == 0:
            print(f'=========iteration:{str(i)}===========')
            for key in optimizers.keys():
                loss = netowrks[key].loss(x_batch, t_batch)
                print(f'{key}:{str(loss)}')

    # グラフの描画
    fig = plt.figure()
    markers = {'SGD': 'o', 'Momentum': 'x', 'AdaGrad': 's', 'Adam': 'D', 'RMSprop': '+'}
    x = np.arange(max_iterations)
    for key in optimizers.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.ylim(0,1)
    plt.legend()
    fig.savefig('../images/optimizer_compare.png')


if __name__ == '__main__':
    main()
