"""
卷积神经网络模型
"""
from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.io import loadmat
import random
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from creat_data import load_data

length = 2048  # 每一份数据的向量长度


class CNN():
    def __init__(self,):
        """
        初始化模型所要用到的参数值
        """
        self.learning_rate = 0.0001
        self.cell_num = 128
        self.iters = 2000
        self.batch_size = 8
        self.x = tf.placeholder(tf.float32, [None, 2048], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')
        self.is_training = tf.placeholder_with_default(
            False, shape=(), name='is_training')
        self.y_hat = self.model(self.x, self.is_training)
        self.loss = self.loss(self.y, self.y_hat)
        self.acc = self.acc(self.y, self.y_hat)
        self.opt = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()

    def model(self, x, is_training):
        """
        定义模型前馈网络结构
        """
        x = tf.reshape(x, shape=[-1, 2048, 1])
        x = tf.layers.dropout(
            x, rate=0.3, training=is_training, name='x_dropout')
        cnn1 = tf.layers.conv1d(
            x, filters=64, kernel_size=125, strides=16, name='conv1', padding='same')
        cnn1 = tf.nn.relu(cnn1, name='relu1')
        cnn1 = tf.layers.max_pooling1d(
            inputs=cnn1, pool_size=2, strides=2, name='pooling1')

        cnn2 = tf.layers.conv1d(
            cnn1, filters=96, kernel_size=15, strides=4, name='conv2', padding='same')
        cnn2 = tf.nn.relu(cnn2, name='relu2')
        cnn2 = tf.layers.max_pooling1d(
            inputs=cnn2, pool_size=2, strides=2, name='pooling2')

        cnn2 = tf.layers.flatten(cnn2, name='flatten')

        fc = tf.layers.dense(cnn2, units=128)
        fc = tf.nn.relu(fc, name='relu3')
        fc = tf.layers.dropout(
            fc, rate=0.3, training=is_training, name='fc_dropout')
        w = tf.Variable(tf.truncated_normal(
            [128, 10], stddev=0.1), name='w', dtype=tf.float32)
        b = tf.Variable(tf.zeros(10), name='b', dtype=tf.float32)
        out = tf.matmul(fc, w) + b
        return out

    def loss(self, y, y_hat):
        """
        定义模型的损失函数
        """
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y))
        return loss

    def acc(self, y, y_hat):
        """
        定义准确率计算函数
        """
        correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy

    def train(self, x_train, y_train, x_test, y_test):
        """
        定义模型训练函数
        """
        self.sess.run(self.init)
        _index_in_epoch = 0
        for i in range(self.iters):
            # 清洗数据
            start = _index_in_epoch
            _index_in_epoch += self.batch_size
            if _index_in_epoch > x_train.shape[0]:
                index = list(range(x_train.shape[0]))
                random.Random(0).shuffle(index)
                x_train = x_train[index]
                y_train = y_train[index]
                start = 0
                _index_in_epoch = self.batch_size
            end = _index_in_epoch
            batch_xs = x_train[start:end]
            batch_ys = y_train[start:end]

            # 定义每次送入网络的数据
            feed_dict1 = {self.is_training: True,
                          self.x: batch_xs, self.y: batch_ys}
            feed_dict2 = {self.is_training: False,
                          self.x: x_test, self.y: y_test}
            feed_dict3 = {self.is_training: False,
                          self.x: batch_xs, self.y: batch_ys}
            self.sess.run(self.opt, feed_dict=feed_dict1)
            if i % 100 == 0:
                """
                每迭代 100 次，打印出训练集与测试集的识别准确率，以及损失函数值
                """
                test_acc = self.sess.run(self.acc, feed_dict=feed_dict2)
                train_acc = self.sess.run(self.acc, feed_dict=feed_dict3)
                loss = self.sess.run(self.loss, feed_dict=feed_dict3)
                print("Iter " + str(i * 1) + ", Minibatch Loss= " + "{:.6f}".format(
                    loss) + ", Training Accuracy= " + "{:.5f}".format(train_acc) +
                    ", Test Accuracy= " + "{:.5f}".format(test_acc))


def main():
    # 导入数据
    x_train, y_train, x_test, y_test = load_data(1772)
    # 定义模型
    cnn = CNN()
    # 训练模型
    cnn.train(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    time0 = datetime.now()
    main()
    time1 = datetime.now()
    print("training time is : ", (time1-time0).seconds)
