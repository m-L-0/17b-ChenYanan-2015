import tensorflow as tf
from PIL import Image
import numpy
import  matplotlib.pyplot as plt
from images_convert import images_convert



# 读取数据
print('loading data...')
data = images_convert()  # 分别读出训练集和验证集

# 转为tfrecord文件
config = ['train', 'validation']
for each in range(len(config)):
    mnist_type = config[each]
    # tfrecord格式文件名
    with tf.python_io.TFRecordWriter('images_' + mnist_type + '.tfrecords') as writer:
        image_path = data[each]
        for num_image in range(image_path.shape[0]):
            image = Image.fromarray(data[each][num_image][1:].reshape(48, 24))
            # print(data[each].images[num_images].reshape(28,28))
            # fig = plt.figure()
            # plotwindow = fig.add_subplot(111)
            # print(image)
            # plt.imshow(image, cmap='gray')    # cmap  图像为灰度图
            # plt.show()
            image_byte = image.tobytes()
            label = data[each][num_image][0]
            example = tf.train.Example(features=tf.train.Features(feature={  
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),  
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_byte]))  
                }))
            writer.write(example.SerializeToString())
print('successful')

# # 解析tfrecord代码       
# filename_queue = tf.train.string_input_producer(['./mnist_test.tfrecords'])    # 创建输入队列，读入流中
# reader = tf.TFRecordReader()  
# _, example = reader.read(filename_queue)  # 返回文件名和文件

# # 取出包含有image 和 label的feature对象
# features = tf.parse_single_example(example,
#                                        features={'label': tf.FixedLenFeature([], tf.int64),
#                                                  'data': tf.FixedLenFeature([], tf.string)})  # 将对应的内存块读为张量流
# image = tf.decode_raw(features['data'], tf.uint8)  # tf.decode_raw可以将字符串解析成图像对应的像素组
# # image = tf.cast(image, tf.float32)
# image = tf.reshape(image, [28, 28, 1])
# label = tf.cast(features['label'], tf.int32)  # 类型转换
# image_batch, label_batch = tf.train.shuffle_batch([image, label],
#                                                       batch_size=1,
#                                                       capacity=100,
#                                                       min_after_dequeue=50)
# image = tf.reshape(image_batch, [28, 28])
# # 开始一个会话
# with tf.Session() as sess:
#     init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#     sess.run(init)
#     # 启动多线程
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     for i in range(3):
#         data, label = sess.run([image_batch, label_batch])  # 在会话中取出image和label
#         result = Image.fromarray(data)  # 这里image是之前提到的
#         result.save(str(i) + '.png')  # 存图片
#         print(label)
#         coord.request_stop()
#         coord.join(threads)