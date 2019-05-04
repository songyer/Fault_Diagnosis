from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.io import loadmat
import random
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import zipfile
from datetime import datetime
from creat_data import load_data

length = 2048  # 每一份数据的向量长度


class DNN():
    def __init__(self,):
        """
        初始化模型所要用到的参数值
        """
        with tf.variable_scope('cnn_gru', reuse=tf.AUTO_REUSE):
            self.learning_rate = 0.0001
            self.cell_num = 64
            self.iters = 2000
            self.batch_size = 16
            self.x = tf.placeholder(tf.float32, [None, 2048], name='x')
            self.y = tf.placeholder(tf.float32, [None, 10], name='y')
            self.is_training = tf.placeholder_with_default(
                False, shape=(), name='is_training')
            self.y_hat, self.lstm_out, self.soft_out = self.model(
                self.x, self.is_training)
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
        with tf.variable_scope('cnn_gru', reuse=tf.AUTO_REUSE):
            x = tf.layers.dropout(
                x, rate=0.3, training=is_training, name='x_dropout')

            fc = tf.layers.dense(x, units=512)
            fc = tf.nn.relu(fc, name='relu1')
            fc = tf.layers.dropout(
                fc, rate=0.3, training=is_training, name='fc1_dropout')

            fc = tf.layers.dense(fc, units=128)
            fc = tf.nn.relu(fc, name='relu2')
            fc = tf.layers.dropout(
                fc, rate=0.3, training=is_training, name='fc2_dropout')

            w = tf.Variable(tf.truncated_normal(
                [128, 10], stddev=0.1), name='w', dtype=tf.float32)
            b = tf.Variable(tf.zeros(10), name='b', dtype=tf.float32)
            out = tf.matmul(fc, w) + b
            soft_out = tf.nn.softmax(out, name='out_soft')

            return out, fc, soft_out

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
        for i in range(self.iters + 1):
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
                          self.x: x_train, self.y: y_train}
            feed_dict2 = {self.is_training: False,
                          self.x: x_test, self.y: y_test}
            feed_dict3 = {self.is_training: False,
                          self.x: x_train, self.y: y_train}
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

        self.sess.close()


def main():
    # 导入数据
    x_train, y_train, x_test, y_test = load_data(1772)
    # 定义模型
    cnn = DNN()
    # 训练模型
    cnn.train(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    time0 = datetime.now()
    main()
    time1 = datetime.now()
    print("training time is : ", (time1-time0).seconds)
