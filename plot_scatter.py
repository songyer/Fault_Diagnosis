
import random
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from time import time
from matplotlib.colors import ListedColormap
import seaborn as sns


def data3():
    data = np.load('pre_data/ pre_data_2.npz')
    train_lstm_out = data['train_lstm_out']
    test_lstm_out = data['test_lstm_out']
    train_soft_out = data['train_soft_out']
    test_soft_out = data['test_soft_out']
    y_train = data['y_train']
    y_test = data['y_test']

    len_x_train = train_lstm_out.shape[0]
    all_data = np.vstack([train_lstm_out, test_lstm_out])
    tsne = TSNE(n_components=2)
    result = tsne.fit_transform(all_data)

    sen_train = result[:len_x_train]
    sen_test = result[len_x_train:]

    color_y_hat = np.zeros([test_soft_out.shape[0], 1])
    for i in range(test_soft_out.shape[0]):
        yy = test_soft_out[i].tolist()
        label = yy.index(max(yy))
        color_y_hat[i] = label

    color_y = np.zeros([y_test.shape[0], 1])
    for i in range(y_test.shape[0]):
        yy = y_test[i].tolist()
        label = yy.index(max(yy))
        color_y[i] = label

    color_x = np.zeros([y_train.shape[0], 1])
    for i in range(y_train.shape[0]):
        yy = y_train[i].tolist()
        label = yy.index(max(yy))
        color_x[i] = label

    ax = plt.subplot(111, axisbg='white')
    cmap_light = ListedColormap(
        ['#9a0eea', '#661aee', '#75fd63', '#fffe40', '#ff9408', '#f8481c', '#a8ff04', '#40a366',  '#99cc04', '#056eee'])  # 给不同区域赋以颜色
    # cmap_bold = ListedColormap(
    #     ['#FF0000', '#003300', '#0000FF'])  # 给不同属性的点赋以颜色
    # plt.pcolor([sen_test[:, 0], sen_test[:, 1]],
    #            C=color_y_hat, cmap=cmap_light)
    # plt.figure(facecolor='#d8dcd6', edgecolor='white',
    #            figsize=(16, 6))

    ax.scatter(sen_test[:, 0], sen_test[:, 1], s=60, c=color_y, edgecolors='none',
               cmap=cmap_light, marker='o')

    ax.scatter(sen_train[:, 0], sen_train[:, 1], s=200, c=color_x,
               cmap=cmap_light, marker='*')
    plt.show()


data3()
