# coding: utf-8
import os, sys
PROJECT_PATH = '/home/ec2-user/ML_Tips/20_DNN/scratch'
sys.path.append(PROJECT_PATH)

import matplotlib.pyplot as plt
import numpy as np

from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD
from common.util import smooth_curve
from dataset.mnist import load_mnist


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    return x_train, t_train, x_test, x_test


def main():
    x_train, t_train, x_test, x_test = get_data()

    # 実験の設定
    train_size = x_train.shape[0]
    batch_size = 128
    max_iterations = 2000

    weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
    optimizer = SGD(lr=0.01)

    networks = {}
    train_loss = {}
    for key, weight_type in weight_init_types.items():
        networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                      output_size=10, weight_init_std=weight_type)
        train_loss[key] = []

    # 訓練の開始
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 実験のパラメータごとに実行
        for key in weight_init_types.keys():
            # 勾配を計算
            grads = networks[key].gradient(x_batch, t_batch)
            # パラメータを更新
            optimizer.update(networks[key].params, grads)

            # 更新されたパラメータで改めて損失を算出
            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        # terminal check
        if i % 100 == 0:
            print(f'=========iteration:{str(i)}===========')
            for key in weight_init_types.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(f'{key}:{str(loss)}')

    # グラフの描画
    fig = plt.figure()
    markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
    x = np.arange(max_iterations)
    for key in weight_init_types.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.ylim(0, 2.5)
    plt.legend()
    fig.savefig('../images/weight_init_compare.png')


if __name__ == '__main__':
    main()
