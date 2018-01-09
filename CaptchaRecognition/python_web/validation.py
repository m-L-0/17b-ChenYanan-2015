import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import os


def validation(image_path):
    # 获取图片，并转换图片格式
    img = Image.open(image_path)
    # resize图片
    new_img = img.resize((48, 32), Image.BILINEAR)
    # 将图片转化为灰度图
    gray = new_img.convert('L')
    # reshape 读取
    gray = np.array(gray, dtype='uint8')
    gray = gray.reshape(32, 48)
    # # 读取
    # fig = plt.figure()
    # plotwindow = fig.add_subplot(111)
    # plt.imshow(gray, cmap='gray')
    # plt.show()
    # reshape 存储
    gray = gray.reshape(1536,)

    # 转换数据类型
    image = gray.astype(float)

    # 数据归一化
    image = image.reshape(-1, 1536)
    image = image / 255

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
    # ckpt = tf.train.latest_checkpoint('./cnn_model')
    # if not ckpt:
    #     if not os.path.exists('./cnn_model'):
    #         os.mkdir('./cnn_model')


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
    with tf.name_scope("accuracy"):
        accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction_1, "float"))
        accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))
        accuracy_3 = tf.reduce_mean(tf.cast(correct_prediction_3, "float"))
        accuracy_4 = tf.reduce_mean(tf.cast(correct_prediction_4, "float"))

        accuracy = (accuracy_1 + accuracy_2 + accuracy_3 + accuracy_4) / 4
        tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all() 
        

    # 保存
    train_writer = tf.summary.FileWriter('graphs/')
    train_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # ckpt = tf.train.latest_checkpoint('./cnn_model')
        # step = 0
        # if ckpt:
        #     saver.restore(sess=sess, save_path=ckpt)
        saver.restore(sess, './cnn_model_1/nums-9049')
            # step = int(ckpt[len(os.path.join(FLAGS.cnn_path, FLAGS.cnn_parameters)) + 1:])

        conv_1 = np.argmax(sess.run(y_conv_1, feed_dict={X_:image, keep_prob: 1.0}), 1)
        conv_2 = np.argmax(sess.run(y_conv_2, feed_dict={X_:image, keep_prob: 1.0}), 1)
        conv_3 = np.argmax(sess.run(y_conv_3, feed_dict={X_:image, keep_prob: 1.0}), 1)
        conv_4 = np.argmax(sess.run(y_conv_4, feed_dict={X_:image, keep_prob: 1.0}), 1)
        if conv_1 == 10:
            sol_1 = ''
        else:
            sol_1 = str(conv_1[0])
        if conv_2 == 10:
            sol_2 = ''
        else:
            sol_2 = str(conv_2[0])
        if conv_3 == 10:
            sol_3 = ''
        else:
            sol_3 = str(conv_3[0])
        if conv_4 == 10:
            sol_4 = ''
        else:
            sol_4 = str(conv_4[0])
        rst = '%s%s%s%s' % (sol_1, sol_2, sol_3, sol_4)

    return rst

