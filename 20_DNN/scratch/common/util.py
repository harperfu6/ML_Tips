# coding: utf-8
import numpy as np
from mypy_extensions import TypedDict


def smooth_curve(x):
    """
    " 損失関数のグラフを滑らかにする
    " http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def im2col(input_data: np.ndarray, filter_h: int, filter_w: int, stride: int=1, pad: int=0) -> np.ndarray:
    """filterしやすいように，input_dataを２次元配列に展開する
    Args:
        input_data (np.ndarray): （データ数，チャンネル，高さ，幅）の４次元配列からなる入力データ
        filter_h (int): フィルターの高さ
        filter_w (int): フィルターの幅
        stride (int): ストライド
        pad (int): パディング
    
    Returns:
        np.ndarray: ２次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    
    # 第２引数の数は，input_dataの次元数に依存
    # データ数，チャンネル，高さ，幅へのパディング数
    # それぞれ（先頭，末尾）へのパディング数
    # 下記例の場合，データ数，チャンネル方向へのパディングはせず，
    # 高さ，幅方向へpad分だけpaddingする
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # データ数，チャンネル，フィルタ高さ，フィルタ幅，出力高さ，出力幅
    # ↓
    # データ数，出力高さ，出力幅，チャンネル，フィルタ高さ，フィルタ幅
    # データ数，出力高さ，出力幅の数でまとめて，２次元配列にする
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col: np.ndarray, input_shape: np.ndarray, filter_h: int, filter_w: int, stride: int=1, pad: int=0) -> np.ndarray:
    """im2colによって２次元配列に変形された入力を元の次元に戻す
    Args:
        col (np.ndarray): im2colによって変形された２次元配列
        input_shape (np.ndarray): （データ数，チャンネル，高さ，幅）の４次元配列からなる入力データの形状
        filter_h (int): フィルターの高さ
        filter_w (int): フィルターの幅
        stride (int): ストライド
        pad (int): パディング
    
    Returns:
        np.ndarray: インプット配列
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]