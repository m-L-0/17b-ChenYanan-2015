import os
import random
import numpy as np
import tensorflow as tf
from read_tfrecord import read_tfrecord
from labels_split import labels_numpy


# 导入数据
# 训练
train_x_1, train_y_1 = read_tfrecord('images_train_1.tfrecords', 4000)
train_x_2, train_y_2 = read_tfrecord('images_train_2.tfrecords', 4000)
train_x_3, train_y_3 = read_tfrecord('images_train_3.tfrecords', 4000)
train_x_4, train_y_4 = read_tfrecord('images_train_4.tfrecords', 4000)
train_x_5, train_y_5 = read_tfrecord('images_train_5.tfrecords', 4000)
train_x_6, train_y_6 = read_tfrecord('images_train_6.tfrecords', 4000)
train_x_7, train_y_7 = read_tfrecord('images_train_7.tfrecords', 4000)
train_x_8, train_y_8 = read_tfrecord('images_train_8.tfrecords', 4000)

# 拼接训练数据
train_x = np.vstack((train_x_1, train_x_2, train_x_3, train_x_4, train_x_5, train_x_6, train_x_7, train_x_8))
train_y = np.vstack((train_y_1, train_y_2, train_y_3, train_y_4, train_y_5, train_y_6, train_y_7, train_y_8))
# 验证
validation_x, validation_y = read_tfrecord('images_validation.tfrecords', 4000)
# 测试
test_x, test_y = read_tfrecord('images_test.tfrecords', 4000)

# 将  _y（labels）转换为one-hot 类型
# 1.转换函数
def dense_to_one_hot(labels_dense, num_classes=11):
    labels_dense = labels_dense.astype(np.uint8)   # 更换数据类型
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# 标签 --> 数组
train_y = labels_numpy(train_y)
validation_y = labels_numpy(validation_y)
test_y = labels_numpy(test_y)

# 热编码
# 将每一列单独转换为热编码
# train
train_y_h_1 = train_y[:, 0]
train_y_h_1 = dense_to_one_hot(train_y_h_1)
train_y_h_2 = train_y[:, 1]
train_y_h_2 = dense_to_one_hot(train_y_h_2)
train_y_h_3 = train_y[:, 2]
train_y_h_3 = dense_to_one_hot(train_y_h_3)
train_y_h_4 = train_y[:, 3]
train_y_h_4 = dense_to_one_hot(train_y_h_4)
# validation
validation_y_h_1 = validation_y[:, 0]
validation_y_h_1 = dense_to_one_hot(validation_y_h_1)
validation_y_h_2 = validation_y[:, 1]
validation_y_h_2 = dense_to_one_hot(validation_y_h_2)
validation_y_h_3 = validation_y[:, 2]
validation_y_h_3 = dense_to_one_hot(validation_y_h_3)
validation_y_h_4 = validation_y[:, 3]
validation_y_h_4 = dense_to_one_hot(validation_y_h_4)
# test
test_y_h_1 = test_y[:, 0]
test_y_h_1 = dense_to_one_hot(test_y_h_1)
test_y_h_2 = test_y[:, 1]
test_y_h_2 = dense_to_one_hot(test_y_h_2)
test_y_h_3 = test_y[:, 2]
test_y_h_3 = dense_to_one_hot(test_y_h_3)
test_y_h_4 = test_y[:, 3]
test_y_h_4 = dense_to_one_hot(test_y_h_4)

# min_next_batch_tfr(随机批次载入数据)    #### 有问题
def min_next_batch_tfr(image, label_1, label_2, label_3, label_4, num=50, count=32000): 
    images = np.zeros((num, 1536))
    labels_1 = np.zeros((num, 11))
    labels_2 = np.zeros((num, 11))
    labels_3 = np.zeros((num, 11))
    labels_4 = np.zeros((num, 11))
    for i in range(num):
        temp = random.randint(0, count-1)
        images[i, :] = image[temp]
        labels_1[i, :] = label_1[temp]
        labels_2[i, :] = label_2[temp]
        labels_3[i, :] = label_3[temp]
        labels_4[i, :] = label_4[temp]

    return images, labels_1, labels_2, labels_3, labels_4

# 参数保存目录
FLAGS = tf.app.flags.FLAGS
# 模型参数
tf.app.flags.DEFINE_string('cnn_path', './cnn_model_z', """存放模型的目录""")

tf.app.flags.DEFINE_string('cnn_parameters', 'nums',"""模型的名称""")

# 构建网络
# 权值初始化
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义卷积层
def conv2d(x, W):
    # 默认 strides[0]=strides[3]=1, strides[1]为x方向步长，strides[2]为y方向步长
    # x 为输入图像， w为卷积核
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')    # 返回feature map

# 定义池化层
def max_pool_2x2(x):
    # x 一般为feature map ksize 为池化窗口的大小 [batch, height, width, channels] , strides为每个维度上滑过的步长
    # [1, stride,stride, 1], 返回一个tensor类型仍为 [batch, height, width, channels]
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#模型文件所在的文件夹，是否存在，如果不存在，则创建文件夹
# ckpt = tf.train.latest_checkpoint(FLAGS.cnn_path)
# if not ckpt:
#     if not os.path.exists(FLAGS.cnn_path):
#         os.mkdir(FLAGS.cnn_path)


# # 声明输入数据的占位符
X_ = tf.placeholder(tf.float32, [None, 1536])
# 声明输出数据的占位符
y_1 = tf.placeholder(tf.float32, [None, 11])
y_2 = tf.placeholder(tf.float32, [None, 11])
y_3 = tf.placeholder(tf.float32, [None, 11])
y_4 = tf.placeholder(tf.float32, [None, 11])

# 把X转为卷积所需要的形式
# 因为图像为灰度图像，所以channel为1 ，-1表示模糊控制的意思，具体是多少图片由tf计算
with tf.name_scope("reshape1"):
    X = tf.reshape(X_, [-1, 32, 48, 1])
# 第一层卷积：3x3×1卷积核32个 [2，2, 1，32],h_conv1.shape=[-1, 32, 48, 32]
with tf.name_scope("conv1_1"):
    W_conv1_1 = weight_variable([2, 2, 1, 16])
    b_conv1_1 = bias_variable([16])
    # relu max(features, 0),负值返回0 并且返回和feature一样的形状的tensor。
    h_conv1_1 = tf.nn.relu(conv2d(X, W_conv1_1) + b_conv1_1)

with tf.name_scope("conv1_2"):
    W_conv1_2 = weight_variable([2, 2, 16, 32])
    b_conv1_2 = bias_variable([32])
    # relu max(features, 0),负值返回0 并且返回和feature一样的形状的tensor。
    h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2)
# 第一个pooling 层[-1, 48, 32, 32]->[-1, 24, 16, 64]
with tf.name_scope("pool1"):
    h_pool1 = max_pool_2x2(h_conv1_2)

# 第二层卷积：3*3×32卷积核64个 [3，3，32，64],h_conv2.shape=[-1, 16, 24, 64]
with tf.name_scope("conv2_1"):
    W_conv2_1 = weight_variable([1, 2, 32, 48])
    b_conv2_1 = bias_variable([48])
    h_conv2_1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1) + b_conv2_1)

with tf.name_scope("conv2_2"):
    W_conv2_2 = weight_variable([2, 2, 48, 64])
    b_conv2_2 = bias_variable([64])
    h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2)

# 第二个pooling 层,[-1, 16, 24, 64]->[-1, 8, 12, 64] 
with tf.name_scope("pool2"):
    h_pool2 = max_pool_2x2(h_conv2_2)

# 第三层卷积：5×5×32*64卷积核96个 [3，3，32，64,96],h_conv2.shape=[-1, 8, 12, 96]
with tf.name_scope("conv3_1"):
    W_conv3_1 = weight_variable([2, 2, 64, 80])
    b_conv3_1 = bias_variable([80])
    h_conv3_1 = tf.nn.relu(conv2d(h_pool2, W_conv3_1) + b_conv3_1)

with tf.name_scope("conv3_2"):
    W_conv3_2 = weight_variable([2, 2, 80, 96])
    b_conv3_2 = bias_variable([96])
    h_conv3_2 = tf.nn.relu(conv2d(h_conv3_1, W_conv3_2) + b_conv3_2)

# 第三个pooling 层,[-1, 8, 12, 96]->[-1, 4, 6, 192] 
with tf.name_scope("pool3"):
    h_pool3 = max_pool_2x2(h_conv3_2)


# [-1, 6, 4, 92]->[-1, 6*4*92],即每个样本得到一个6*3*92维的样本
with tf.name_scope("flatting"):
    h_pool3_flat = tf.reshape(h_pool3, [-1, 4*6*96])

# 全连接
# fc1
# w_fc1的第一维度表示第二层卷积层的输出，大小为7*7带有64个过滤图，第二个参数是层中的神经元数量，我们可自由设置。
with tf.name_scope("fc1"):
    W_fc1 = weight_variable([4*6*96, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
# tf.nn.dropout是TensorFlow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层。
# 以keep_prob的概率值决定是否被抑制，若抑制则神经元为0，若不被抑制，则神经元输出值y y∗=1keep_prob
with tf.name_scope("dropout"):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层1
with tf.name_scope("fcc1"):
    W_fc2_1 = weight_variable([1024, 11])
    b_fc2_1 = bias_variable([11])
    # softmax分类函数 输出二分类或多分类任务中某一类的概率，将输入排序，并转换为概率表示
    y_conv_1 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2_1) + b_fc2_1)

# 输出层2
with tf.name_scope("fcc2"):
    W_fc2_2 = weight_variable([1024, 11])
    b_fc2_2 = bias_variable([11])
    # softmax分类函数 输出二分类或多分类任务中某一类的概率，将输入排序，并转换为概率表示
    y_conv_2 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2_2) + b_fc2_2)

# 输出层3
with tf.name_scope("fcc3"):
    W_fc2_3 = weight_variable([1024, 11])
    b_fc2_3 = bias_variable([11])
    # softmax分类函数 输出二分类或多分类任务中某一类的概率，将输入排序，并转换为概率表示
    y_conv_3 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2_3) + b_fc2_3)

# 输出层4
with tf.name_scope("fcc4"):
    W_fc2_4 = weight_variable([1024, 11])
    b_fc2_4 = bias_variable([11])
    # softmax分类函数 输出二分类或多分类任务中某一类的概率，将输入排序，并转换为概率表示
    y_conv_4 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2_4) + b_fc2_4)


# 训练和评估
# 1.损失函数：cross_entropy  交叉熵代价函数
with tf.name_scope("cross_entropy"):
    cross_entropy_1 = -tf.reduce_sum(y_1*tf.log(y_conv_1), name='cross_entropy_1')
    cross_entropy_2 = -tf.reduce_sum(y_2*tf.log(y_conv_2), name='cross_entropy_2')
    cross_entropy_3 = -tf.reduce_sum(y_3*tf.log(y_conv_3), name='cross_entropy_3')
    cross_entropy_4 = -tf.reduce_sum(y_4*tf.log(y_conv_4), name='cross_entropy_4')
    cross_entropy = tf.reduce_sum([cross_entropy_1, cross_entropy_2, cross_entropy_3, cross_entropy_4], name='cross_entropy')
    # 变量的曲线图
    tf.summary.scalar('loss_1', cross_entropy_1)  
    tf.summary.scalar('loss_2', cross_entropy_2)
    tf.summary.scalar('loss_3', cross_entropy_3)
    tf.summary.scalar('loss_4', cross_entropy_4)
    tf.summary.scalar('loss', cross_entropy)
    merged = tf.summary.merge_all() 
# 2.优化函数：AdamOptimizer  实现adam算法的优化器，是一个寻找全局最优点的优化算法，可以控制学习速度  1e-4 为参数epsilon学习率的值
with tf.name_scope("Adam"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 3.预测准确结果统计
# 预测值中最大值（１）即分类结果，是否等于原始标签中的（１）的位置。argmax()取最大值所在的下标
with tf.name_scope("prediction"):
    correct_prediction_1 = tf.equal(tf.argmax(y_conv_1, 1), tf.argmax(y_1, 1))
    correct_prediction_2 = tf.equal(tf.argmax(y_conv_2, 1), tf.argmax(y_2, 1))
    correct_prediction_3 = tf.equal(tf.argmax(y_conv_3, 1), tf.argmax(y_3, 1))
    correct_prediction_4 = tf.equal(tf.argmax(y_conv_4, 1), tf.argmax(y_4, 1))
with tf.name_scope("accuracy_total"):
    accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction_1, "float"))
    accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))
    accuracy_3 = tf.reduce_mean(tf.cast(correct_prediction_3, "float"))
    accuracy_4 = tf.reduce_mean(tf.cast(correct_prediction_4, "float"))

    accuracy_total = (accuracy_1 + accuracy_2 + accuracy_3 + accuracy_4) / 4
    tf.summary.scalar('accuracy', accuracy_total)
    merged = tf.summary.merge_all() 
with tf.name_scope('accurancy_byte'):
    y_conva = tf.concat([y_conv_1, y_conv_2], 1) 
    y_convb = tf.concat([y_conv_3, y_conv_4], 1)
    y_conv = tf.concat([y_conva, y_convb], 1)
    predict = tf.reshape(y_conv, [-1, 4, 11])
    max_idx_p = tf.argmax(predict, 2)

    y_conva = tf.concat([y_1, y_2], 1) 
    y_convb = tf.concat([y_3, y_4], 1)
    y_conv = tf.concat([y_conva, y_convb], 1)
    initail = tf.reshape(y_conv, [-1, 4, 11])
    max_idx_l = tf.argmax(initail, 2)
    # max_idx_l = tf.argmax(tf.reshape(y_, [-1, 4, 11]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    accuracy_byte = tf.reduce_mean(tf.cast(tf.cast(tf.reduce_mean(tf.cast(correct_pred,tf.float32),1),tf.int64),tf.float32))   

# 保存
train_writer = tf.summary.FileWriter('graphs/')
train_writer.add_graph(tf.get_default_graph())
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # ckpt = tf.train.latest_checkpoint(FLAGS.cnn_path)
    # step = 0
    # if ckpt:
    # saver.restore(sess=sess, save_path=ckpt)
    # saver.restore(sess, './cnn_model_1/nums-9049')
    # saver = tf.train.import_meta_graph('./cnn_model_1/nums-9049.meta')
    saver.restore(sess, './cnn_model_2/nums-9250')
    # step = int(ckpt[len(os.path.join(FLAGS.cnn_path, FLAGS.cnn_parameters)) + 1:])

    # 开启线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 训练
    # for i in range(12000):   ## 20000
    #     batch_t = min_next_batch_tfr(train_x, train_y_h_1, train_y_h_2, train_y_h_3, train_y_h_4)
    #     batch_v = min_next_batch_tfr(validation_x, validation_y_h_1, validation_y_h_2, validation_y_h_3, validation_y_h_4, num=50, count=4000)
    #     if i % 100 == 0:   ## 1000
    #         summary, validation_accuracy_1, validation_accuracy_2, validation_accuracy_3, validation_accuracy_4, validation_accuracy_total, validation_accuracy_byte, loss = sess.run([merged, accuracy_1, accuracy_2, accuracy_3, accuracy_4, accuracy_total, accuracy_byte, cross_entropy], feed_dict={
    #             X_: batch_v[0], y_1: batch_v[1], y_2: batch_v[2], y_3: batch_v[3], y_4: batch_v[4], keep_prob: 1})
    #         print("step %d, validating accuracy %g %g %g %g %g %g, loss is %g" % (i, validation_accuracy_1, validation_accuracy_2, validation_accuracy_3, validation_accuracy_4, validation_accuracy_total, validation_accuracy_byte, loss))
    #         # saver.save(sess, './cnn_model/cnn_model.ckpt')
    #         ckptname = os.path.join(FLAGS.cnn_path, FLAGS.cnn_parameters)
    #         # print(validation_accuracy*10000)
            
    #         if validation_accuracy_total > 0.83:
    #             saver.save(sess, ckptname, global_step=int(validation_accuracy_total*10000))
    #         # saver.save(sess, ckptname, global_step=int(validation_accuracy*10000))
    #         train_writer.add_summary(summary, i)
    #     train_step.run(feed_dict={X_: batch_t[0], y_1: batch_t[1], y_2: batch_t[2], y_3: batch_t[3], y_4: batch_t[4], keep_prob: 0.5})
        
    # 测试
    # print("test accuracy_total %g" % accuracy_total.eval(feed_dict={
    #     X_: test_x, y_1: test_y_h_1, y_2: test_y_h_2, y_3: test_y_h_3, y_4: test_y_h_4, keep_prob: 1.0}))
    # # print()
    # batch_t = min_next_batch_tfr(test_x, test_y_h_1, test_y_h_2, test_y_h_3, test_y_h_4, num=50, count=4000)
    test_accuracy_1, test_accuracy_2, test_accuracy_3, test_accuracy_4, test_accuracy_total, test_accuracy_byte = sess.run([accuracy_1, accuracy_2, accuracy_3, accuracy_4, accuracy_total, accuracy_byte], feed_dict={
                # X_: batch_t[0], y_1: batch_t[1], y_2: batch_t[2], y_3: batch_t[3], y_4: batch_t[4], keep_prob: 1.0})
                X_: test_x, y_1: test_y_h_1, y_2: test_y_h_2, y_3: test_y_h_3, y_4: test_y_h_4, keep_prob: 1.0})
    print("testing accuracy %g %g %g %g %g %g" % (test_accuracy_1, test_accuracy_2, test_accuracy_3, test_accuracy_4, test_accuracy_total, test_accuracy_byte))
    coord.request_stop()  
    coord.join(threads)





testing accuracy 0.92625 0.87175 0.846 0.8785 0.880625 0.65925testing accuracy 0.92625 0.87175 0.846 0.8785 0.880625 0.65925testing accuracy 0.92625 0.87175 0.846 0.8785 0.880625 0.65925testing accuracy 0.92625 0.87175 0.846 0.8785 0.880625 0.65925testing accuracy 0.92625 0.87175 0.846 0.8785 0.880625 0.65925testing accuracy 0.92625 0.87175 0.846 0.8785 0.880625 0.65925


    testing accuracy 0.92625 0.87175 0.846 0.8785 0.880625 0.65925testing accuracy 0.92625 0.87175 0.846 0.8785 0.880625 0.65925