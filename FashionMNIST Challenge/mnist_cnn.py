# -*- coding: utf-8 -*-
# name:chenyanan

import os
import random
import numpy as np
import tensorflow as tf
from read_tfrecord import read_tfrecord

# 一、环境设置
# 设置按需使用GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)   # 启动计算图

# 二、导入数据
train_x, train_y = read_tfrecord('mnist_train.tfrecords', 55000)
test_x, test_y = read_tfrecord('mnist_test.tfrecords', 10000)

# 三、数据转换
# 将  _y（labels）转换为one-hot 类型
# 1.转换函数
def dense_to_one_hot(labels_dense, num_classes=10):
    labels_dense = labels_dense.astype(np.uint8)   # 转换数据类型
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
# 2.函数实现
train_y = dense_to_one_hot(train_y, 10)
test_y = dense_to_one_hot(test_y, 10)
print(train_y.shape)
print(test_y.shape)


# min_next_batch_tfr(随机批次载入数据)
def min_next_batch_tfr(image, label, num=50): 
    images = np.zeros((num, 784))
    labels = np.zeros((num, 10))
    for i in range(num):
        temp = random.randint(0, 54999)
        images[i, :] = image[temp]
        labels[i, :] = label[temp]

    return images, labels


# 参数保存目录
FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_string('cnn_path', './cnn_model', """存放模型的目录""")

tf.app.flags.DEFINE_string('cnn_parameters', 'mnist',"""模型的名称""")


# 五、构建网络, 写为函数，可以在网络中方便调用

# 5.1 权值初始化，随机选取，返回的值中不会偏离均值的两倍差
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 5.2 偏值的初始化
def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 5.3定义卷积层
def conv2d(x, W):
    # 默认 strides[0]=strides[3]=1, strides[1]为x方向步长，strides[2]为y方向步长
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')    # 返回feature map

# 5.4定义池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#模型文件所在的文件夹，是否存在，如果不存在，则创建文件夹
ckpt = tf.train.latest_checkpoint(FLAGS.cnn_path)
if not ckpt:
    if not os.path.exists(FLAGS.cnn_path):
        os.mkdir(FLAGS.cnn_path)

# X_ 是手写图像的像素值， y是图像对应的标签
X_ = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 把X转为卷积所需要的形式
# 输入层为输入的灰度图像尺寸
X = tf.reshape(X_, [-1, 28, 28, 1])
# 第一层卷积：5×5×1卷积核32个 [5，5，1，32],h_conv1.shape=[-1, 28, 28, 32]
W_conv1 = weight_variable([5, 5, 1, 32])  # 卷积核的大小、深度和数量
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)

# 第一个池化层[-1, 28, 28, 32]->[-1, 14, 14, 32]
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积：5×5×32卷积核64个 [5，5，32，64],h_conv2.shape=[-1, 14, 14, 64]
W_conv2 = weight_variable([5, 5, 32, 64])  # 卷积核的大小、深度和数量
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 第二个池化层,[-1, 14, 14, 64]->[-1, 7, 7, 64] 
h_pool2 = max_pool_2x2(h_conv2)

# flatten层，[-1, 7, 7, 64]->[-1, 7*7*64],即每个样本得到一个7*7*64维的样本
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# 5.5 全连接层，隐藏层节点为1024个
# 权值初始化
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 全连接隐藏层/输出层
# 为了防止网络出现过拟合的情况,对全连接隐藏层进行 Dropout(正则化)处理,在训练过程中随机的丢弃部分
# 节点的数据来防止过拟合.Dropout同把节点数据设置为0来丢弃一些特征值,仅在训练过程中,
# 预测的时候,仍使用全数据特征
# 传入丢弃节点数据的比例
# dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
keep_prob = tf.placeholder(tf.float32)   # 正则化，丢弃比例
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层与全连接隐藏层之间
# 隐藏层与输出层权重初始化
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# 激活后的输出
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 六、训练和评估

# 1.损失函数：cross_entropy  交叉熵代价函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# 2.优化函数：AdamOptimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 3.预测准确结果统计
# 预测值中最大值（１）即分类结果，是否等于原始标签中的（１）的位置。argmax()取最大值所在的下标
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# 准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 保存
saver = tf.train.Saver(max_to_keep=2)

# 全局初始化
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(FLAGS.cnn_path, sess.graph)

ckpt = tf.train.latest_checkpoint(FLAGS.cnn_path)
step = 0
if ckpt:
    saver.restore(sess=sess, save_path=ckpt)
    step = int(ckpt[len(os.path.join(FLAGS.cnn_path, FLAGS.cnn_parameters)) + 1:])

# 开启线程
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# 训练
for i in range(20000):   ## 20000
    batch = min_next_batch_tfr(train_x, train_y, 50)
    if i % 1000 == 0:   ## 1000
        train_accuracy = accuracy.eval(feed_dict={
            X_: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        ckptname=os.path.join(FLAGS.cnn_path, FLAGS.cnn_parameters)
        saver.save(sess,ckptname,global_step=i)
    train_step.run(feed_dict={X_: batch[0], y_: batch[1], keep_prob: 0.5})
    
# 测试
for j in range(200):
    testSample_start = j * 50      
    testSample_end = (j + 1) * 50  
    print('test %d accuracy %g' % (j, accuracy.eval(session=sess, feed_dict={
        X_: test_x[testSample_start:testSample_end],
        y_: test_y[testSample_start:testSample_end], keep_prob: 1.0})))  # 喂数据
coord.request_stop() 
coord.join(threads)
