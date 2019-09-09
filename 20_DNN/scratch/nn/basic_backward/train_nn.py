# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    return x_train, t_train, x_test, t_test

def init_network():
    return TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

def main():
    x_train, t_train, x_test, t_test = get_data()
    network = init_network()

    iters_num = 10000 # epoch_num = iters_num / iter_per_epoch
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1) # 少なくとも１回は回す

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配の計算
        # grad = network.numerical_gradient(x_batch, t_batch) # 勾配の数値微分チェック
        grad = network.gradient(x_batch, t_batch)

        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 損失の計算
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # １エポックの終了
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print('train acc, test acc | {}, {}'.format(str(train_acc), str(test_acc)))

    # 学習経過をグラフに描画
    fig = plt.figure()
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    # plt.show()
    fig.savefig('../images/accuracy_two_layer.png')

if __name__ == '__main__':
    main()