import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# 将训练集和测试集分为训练集、验证集和测试集
data = input_data.read_data_sets('data_initial',           # 读取的时候规定好dtype
                                    dtype=tf.uint8,
                                    reshape=False,
                                ) 


#  转为tfrecord文件
# 路径
config = [{'dir': 'data_initial/train/', 'type': 'train'},
          {'dir': 'data_initial/validation/', 'type': 'validation'},
          {'dir': 'data_initial/test/', 'type': 'test'},
          ]

for each in range(len(config)):
    mnist_dir = config[each]['dir']
    mnist_type = config[each]['type']
    # tfrecord格式文件名
    with tf.python_io.TFRecordWriter('mnist_' + mnist_type + '.tfrecords') as writer:
        image_path = data[each].images
        for num_images in range(image_path.shape[0]):
            image_byte = data[each].images[num_images].tobytes()
            label = data[each].labels[num_images]
            example = tf.train.Example(features=tf.train.Features(feature={  
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),  
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_byte]))  
                }))
            writer.write(example.SerializeToString())
print('successful')