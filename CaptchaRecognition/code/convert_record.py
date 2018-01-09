import tensorflow as tf
from PIL import Image
import numpy
import  matplotlib.pyplot as plt
from images_convert import images_convert



# 读取数据
print('loading data...')
data = images_convert()


# 转为tfrecord文件
config = ['train_1', 'train_2', 'train_3',  'train_4', 'train_5', 'train_6', 'train_7','train_8', 'validation', 'test']
for each in range(len(config)):
    mnist_type = config[each]
    # tfrecord格式文件名
    with tf.python_io.TFRecordWriter('images_' + mnist_type + '.tfrecords') as writer:
        image_path = data[each]
        for num_image in range(image_path.shape[0]):
            img = data[each][num_image][1:]
            img = img.astype('uint8')
            image = Image.fromarray(img.reshape(32, 48))
            # fig = plt.figure()
            # plotwindow = fig.add_subplot(111)
            # plt.imshow(image, cmap='gray')    # cmap  图像为灰度图
            # label = data[each][num_image][0]
            # print(label)
            # plt.show()
            image_byte = image.tobytes()
            label = data[each][num_image][0]
            example = tf.train.Example(features=tf.train.Features(feature={  
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),  
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_byte]))  
                }))
            writer.write(example.SerializeToString())
print('successful')

