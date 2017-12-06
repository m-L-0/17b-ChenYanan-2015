import os
import random
import numpy as np
import tensorflow as tf
from read_tfrecord import read_tfrecord

# 设置按需使用GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)   # 启动计算图


# 导入数据
# 训练
train_x, train_y = read_tfrecord('images_train.tfrecords', 10856)
# 测试
validation_x, validation_y = read_tfrecord('images_validation.tfrecords', 3619)

# 将  _y（labels）转换为one-hot 类型
# 1.转换函数
def dense_to_one_hot(labels_dense, num_classes=100):
    labels_dense = labels_dense.astype(np.uint8)   # 更换数据类型
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
# 2.函数实现
train_y = dense_to_one_hot(train_y, 100)
validation_y = dense_to_one_hot(validation_y, 100)

# min_next_batch_tfr(随机批次载入数据)
def min_next_batch_tfr(image, label, num=50): 
    images = np.zeros((num, 1152))
    labels = np.zeros((num, 100))
    for i in range(num):
        temp = random.randint(0, 10855)
        images[i, :] = image[temp]
        labels[i, :] = label[temp]

    return images, labels

# 参数保存目录
FLAGS = tf.app.flags.FLAGS
# 模型参数
tf.app.flags.DEFINE_string('cnn_path', './cnn_model', """存放模型的目录""")

tf.app.flags.DEFINE_string('cnn_parameters', 'mnist',"""模型的名称""")


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
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#模型文件所在的文件夹，是否存在，如果不存在，则创建文件夹
ckpt = tf.train.latest_checkpoint(FLAGS.cnn_path)
if not ckpt:
    if not os.path.exists(FLAGS.cnn_path):
        os.mkdir(FLAGS.cnn_path)

# 声明输入数据的占位符
X_ = tf.placeholder(tf.float32, [None, 1152])
# 声明输出数据的占位符
y_ = tf.placeholder(tf.float32, [None, 100])

# 把X转为卷积所需要的形式
# 因为图像为灰度图像，所以channel为1 ，-1表示模糊控制的意思，具体是多少图片由tf计算
X = tf.reshape(X_, [-1, 48, 24, 1])
# 第一层卷积：3x3×1卷积核32个 [3，3, 1，32],h_conv1.shape=[-1, 48, 24, 32]
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
# relu max(features, 0),负值返回0 并且返回和feature一样的形状的tensor。
h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
# 第一个pooling 层[-1, 48, 24, 32]->[-1, 24, 12, 32]
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积：5×5×32卷积核64个 [3，3，32，64],h_conv2.shape=[-1, 24, 12, 64]
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# 第二个pooling 层,[-1, 24, 12, 64]->[-1, 12, 6, 64] 
h_pool2 = max_pool_2x2(h_conv2)

# 第三层卷积：5×5×32*64卷积核96个 [3，3，32，64,96],h_conv2.shape=[-1, 12, 6, 96]
W_conv3 = weight_variable([3, 3, 64, 96])
b_conv3 = bias_variable([96])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

# 第三个pooling 层,[-1, 12, 6, 96]->[-1, 6, 3, 96] 
h_pool3 = max_pool_2x2(h_conv3)


# [-1, 6, 3, 92]->[-1, 6*3*92],即每个样本得到一个6*3*92维的样本
h_pool3_flat = tf.reshape(h_pool3, [-1, 6*3*96])

# 全连接
# fc1
# w_fc1的第一维度表示第二层卷积层的输出，大小为7*7带有64个过滤图，第二个参数是层中的神经元数量，我们可自由设置。
W_fc1 = weight_variable([6*3*96, 512])
b_fc1 = bias_variable([512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
# tf.nn.dropout是TensorFlow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层。
# 以keep_prob的概率值决定是否被抑制，若抑制则神经元为0，若不被抑制，则神经元输出值y y∗=1keep_prob
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([512, 100])
b_fc2 = bias_variable([100])
# softmax分类函数 输出二分类或多分类任务中某一类的概率，将输入排序，并转换为概率表示
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练和评估
# 1.损失函数：cross_entropy  交叉熵代价函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# 2.优化函数：AdamOptimizer  实现adam算法的优化器，是一个寻找全局最优点的优化算法，可以控制学习速度  1e-4 为参数epsilon学习率的值
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 3.预测准确结果统计
# 预测值中最大值（１）即分类结果，是否等于原始标签中的（１）的位置。argmax()取最大值所在的下标
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 保存
saver = tf.train.Saver(max_to_keep=2)

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
# for i in range(1000):   ## 20000
#     batch = min_next_batch_tfr(train_x, train_y, 50)
#     if i % 100 == 0:   ## 1000
#         train_accuracy, loss = sess.run([accuracy, cross_entropy], feed_dict={
#             X_: batch[0], y_: batch[1], keep_prob: 1})
#         print("step %d, validating accuracy %g, loss is %g" % (i, train_accuracy, loss))
#         ckptname=os.path.join(FLAGS.cnn_path, FLAGS.cnn_parameters)
#         saver.save(sess,ckptname,global_step=i)
#     train_step.run(feed_dict={X_: batch[0], y_: batch[1], keep_prob: 0.5})
    
# 测试
print("test accuracy %g" % accuracy.eval(feed_dict={
    X_: validation_x, y_: validation_y, keep_prob: 1.0}))

# prediction
y_conv = np.argmax(y_conv.eval(feed_dict={X_: validation_x, y_: validation_y, keep_prob: 1.0}), 1)
y_conv = y_conv.reshape(-1, 1)
coord.request_stop() 
coord.join(threads)

# validation
validation_y = np.argmax(validation_y, 1)
validation_y = validation_y.reshape(-1)

# 拼接  validation + prediction
test = np.column_stack((validation_y, y_conv))


# 按第一列进行排序
test = test[np.lexsort(test[:, ::-1].T)] 
# 计算召回率
for count in range(34):
    for i in range(test.shape[0]):
        j = test[i][0]  # 找出validation 的 label
        if i != 0:
            j1 = test[i-1][0]
            if j1 != j:
                test_temp = test[:i]
                correct_prediction = np.equal(test_temp[:, 0], test_temp[:, 1])
                accuracy = np.mean(correct_prediction)
                accuracy = accuracy.astype('float32')
                if j1 > 10:
                    j1 = chr(j1)
                    print('%s的召回率为%f'%(j1, accuracy))
                else:
                    print('%d的召回率为%f'%(j1, accuracy))

                test = test[i:]
                break

# z的召回率
test_temp = test
correct_prediction = np.equal(test_temp[:, 0], test_temp[:, 1])
accuracy = np.mean(correct_prediction)
accuracy = accuracy.astype('float32')
print('z的召回率为%f'% accuracy)