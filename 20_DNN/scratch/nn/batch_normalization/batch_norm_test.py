# coding: utf-8
# 重みの初期値を正しく設定することで学習がスムーズに進むが，何が正しいかは難しい
# そこで，各層の学習後に無理やり分布を分散させてあげるという考え方
# - 学習を早く進行させることができる
# - 初期値にそれほど依存しない
# - 過学習を抑制する（Dropoutなどの必要性を減らす）

import os, sys
PROJECT_PATH = '/home/ec2-user/ML_Tips/20_DNN/scratch'
sys.path.append(PROJECT_PATH)

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple, Optional

from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    return x_train, t_train, x_test, x_test

def main() -> None:
    x_train, t_train, x_test, x_test = get_data()

    # 学習データを削減
    x_train = x_train[:1000]
    t_train = t_train[:1000]

    # 実験の設定
    max_epochs = 20
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01

    # 学習とグラフの描画
    weight_scale_list = np.logspace(0, -4, num=16)
    x = np.arange(max_epochs)

    fig = plt.figure()
    for i, w in enumerate(weight_scale_list):
        print( "============== " + str(i+1) + "/16" + " ==============")
        train_acc_list, bn_train_acc_list = __train(x_train, t_train, 
                                                    train_size, max_epochs, batch_size,
                                                    learning_rate, w)

        plt.subplot(4,4,i+1)
        plt.title('W:' + str(w))
        if i == 15:
            plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
            plt.plot(x, train_acc_list, linestyle='--', label='Normal(without BatchNorm)', markevery=2)
        else:
            plt.plot(x, bn_train_acc_list, markevery=2)
            plt.plot(x, train_acc_list, linestyle='--', markevery=2)
        
        plt.ylim(0, 1.0)
        if i % 4:
            plt.yticks([])
        else:
            plt.ylabel('accuracy')
        if i < 12:
            plt.xticks([])
        else:
            plt.xlabel('epochs')
        plt.legend(loc='lower right')
    
    fig.savefig(os.path.join(PROJECT_PATH, 'nn', 'images', 'batch_norm_test.png'))



def __train(x_train: np.ndarray, t_train: np.ndarray,
            train_size: int, max_epochs: int, batch_size: int,
            learning_rate: float, weight_init_std: float
            ) -> Tuple[List[float], List[float]]:

    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                     weight_init_std=weight_init_std, use_batchNorm=True)
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                     weight_init_std=weight_init_std, use_batchNorm=False)

    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list
    
if __name__ == '__main__':
    main()