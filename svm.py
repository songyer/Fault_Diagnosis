from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.io import loadmat
import random
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import pywt
import jieba


def load_data():
    data_train = np.load('data/ 1750_n1.npz')
    x_train = data_train['data_x']
    y_train = data_train['data_y']
    data_test = np.load('data/ 1750_test.npz')
    x_test = data_test['data_x']
    y_test = data_test['data_y']
    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = load_data()

    # pca = PCA(n_components=128)
    # x_train = pca.fit_transform(x_train)
    # x_test = pca.fit_transform(x_test)

    coeffs = pywt.wavedec(x_train[0], 'bior3.7', level=5)
    print(round(coeffs[0], 3))

    # y_tra = []
    # y_te = []
    # for i in y_train:
    #     y_tra.append(np.argmax(i))
    # for i in y_test:
    #     y_te.append(np.argmax(i))
    # y_tra = np.array(y_tra)
    # y_te = np.array(y_te)

    # model = SVC()
    # model.fit(x_train, y_tra)
    # y_hat = model.predict(x_test)

    # acc = accuracy_score(y_te, y_hat)
    # print('acc:', acc)


def cut1():
    a = '影响版本也许兼容低版本低版本版本功能'
    b = jieba.cut(a)
    print(list(b))


if __name__ == '__main__':
    main()
    # cut1()
