from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.io import loadmat
import random
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import zipfile
import warnings

length = 2048  # 每一份数据的向量长度
multiple = 1  # 重叠采样的的分数
# num_train = 11


def load_mat(file_path, label, multiple):
    """导入 .mat 文件"""
    data_x = np.zeros((0, 2048))
    data_y = np.zeros((0, 10))
    mat_dict = loadmat(file_path)
    fliter_i = filter(lambda x: 'DE_time' in x, mat_dict.keys())
    fliter_list = [item for item in fliter_i]
    key = fliter_list[0]
    time_series = mat_dict[key][:, 0]
    step = int(length / multiple)
    for i in range(1):
        start = step * i
        new_time_serices = time_series[start:]
        idx_last = -(new_time_serices.shape[0] % length)
        clips = new_time_serices[:idx_last].reshape(-1, length)
        n = clips.shape[0]
        data_x = np.vstack((data_x, clips))
        y = np.tile(label, (n, 1))
        data_y = np.vstack((data_y, y))
    index = list(range(data_x.shape[0]))
    random.Random(0).shuffle(index)
    data_x = data_x[index]
    data_y = data_y[index]

    return data_x, data_y


def load_data(speed, num_train=50, multiple=1):
    """构造训练数据集和测试数据集"""
    falut_class = [
        '0.007-Ball',
        '0.007-InnerRace',
        '0.007-OuterRace6',
        '0.014-Ball',
        '0.014-InnerRace',
        '0.014-OuterRace6',
        '0.021-Ball',
        '0.021-InnerRace',
        '0.021-OuterRace6',
        'Normal'
    ]
    falut_labels = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    file_path = []
    for i in range(len(falut_class)):
        temp = 'CWRU/'+str(speed)+'/'+falut_class[i]+'.mat'
        file_path.append(temp)

    x_train = np.zeros((0, 2048))
    y_train = np.zeros((0, 10))
    x_test = np.zeros((0, 2048))
    y_test = np.zeros((0, 10))

    for i in range(len(falut_class)):
        x, y = load_mat(file_path[i], falut_labels[i], multiple)

        x_train = np.vstack((x_train, x[:num_train]))
        y_train = np.vstack((y_train, y[:num_train]))

        x_test = np.vstack((x_test, x[num_train:]))
        y_test = np.vstack((y_test, y[num_train:]))

    # 洗牌操作
    index = list(range(x_train.shape[0]))
    random.Random(0).shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]
    print(x_train.shape)
    print(y_train.shape)

    index = list(range(x_test.shape[0]))
    random.Random(0).shuffle(index)
    x_test = x_test[index]
    y_test = y_test[index]
    print(x_test.shape)
    print(y_test.shape)

    return x_train, y_train, x_test, y_test


def wgn(x, snr):
    """计算信噪比函数"""
    snr = 10 ** (snr / 10)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    y = x + np.random.randn(len(x)) * np.sqrt(npower)
    return y


def add_noise(data):
    """添加噪声函数"""
    data_noise = np.zeros((0, 2048))
    for i in data:
        b = wgn(i, 0)
        # plt.plot(i)
        # plt.plot(b)
        # plt.show()
        b = b.reshape(-1, 2048)
        data_noise = np.vstack((data_noise, b))
    return data_noise


# def main():
#     for i in range(1, 16):
#         x_train, y_train, _, _ = load_data(1772, i)
#         print(x_train.shape)
#         print(y_train.shape)
#         np.savez('data/ 1772_n'+str(i), data_x=x_train, data_y=y_train)

#     speeds = [1772, 1750, 1730]
#     for speed in speeds:
#         _, _, x_test, y_test = load_data1(speed, 0)
#         # x_test = add_noise(x_test)
#         np.savez('data/ '+str(speed)+'_test', data_x=x_test, data_y=y_test)


# if __name__ == '__main__':
#     main()
