# -*- coding: utf-8 -*-
# name:chenyanan

import tensorflow as tf
from PIL import Image
import numpy as np
from read_tfrecord import read_tfrecord


def read_tfrecord(config_dir, num = 1):   
    # 读取tfrecord代码      
    filename_queue = tf.train.string_input_producer([config_dir])    # 创建输入队列，读入流中
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)  # 返回文件名和文件

    # 取出包含有image 和 label的feature对象
    features = tf.parse_single_example(example,
                                        features={'label': tf.FixedLenFeature([], tf.int64),
                                                    'data': tf.FixedLenFeature([], tf.string)})  # 将对应的内存块读为张量流
    image = tf.decode_raw(features['data'], tf.uint8)  # tf.decode_raw可以将字符串解析成图像对应的像素组
    image = tf.cast(image, tf.float32)    # 解码之后转数据类型 
    image = tf.reshape(image, [28, 28])
    label = tf.cast(features['label'], tf.int32)  # 类型转换
    # 随机读取数据，验证图片对应正确性
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=1,
                                                        capacity=100,
                                                        min_after_dequeue=50)

    # 开始一个会话
    with tf.Session() as sess:
        exm_images = np.zeros((num, 784))
        exm_labels = np.zeros((num, 1))

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        # 启动多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for count in range(num):

            image, label = sess.run([image_batch, label_batch])  # 在会话中取出image和label
            img = image.reshape([28, 28])  # 这里要reshape因为默认一个批次处理的数据会外层嵌套一层
            img = img.astype(np.uint8)  # PIL保存时，必须是整数
            if num == 1:
                coord.request_stop()  
                coord.join(threads)
                return img, label       # 在进行图片验证时使用
            else:
                image = image.reshape(784)
                # for i in range(784):
                #     # if image[i] > 127:
                #     #     image[i] = 1
                #     # else:
                #     #     image[i] = 0     # 改进 ： 会让正确率提高5% 左右
                image = image / 255
                exm_images[count, :] = image
                exm_labels[count, :] = label
                if count % 10000 == 0:
                    print(count)
        coord.request_stop()  
        coord.join(threads)
    return exm_images, exm_labels        # 在进行算法读取数据矩阵时使用
