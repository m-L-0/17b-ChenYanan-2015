import tensorflow as tf
import numpy as np


def labels_csv_numpy():
    g = tf.Graph()
    with g.as_default():
        # 生成“文件名队列”
        filenames = tf.train.match_filenames_once('../data/captcha/labels/*.csv')
        filename_queue = tf.train.string_input_producer(filenames)

        # 读取数据     
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)

        # 解码csv文件  record_default的值根据实际情况写
        # decoded = tf.decode_csv(value, record_defaults=[[0], [0]])
        decoded = tf.decode_csv(value, record_defaults=[['string'], ['int']])

        # 创建“样本队列”  这里容量与类型需要根据实际情况填写 
        example_queue = tf.FIFOQueue(5, tf.string)
        # 入队操作    
        enqueue_example = example_queue.enqueue([decoded])
        # 出队操作   根据需要也可以dequeue_many 
        dequeue_example = example_queue.dequeue()

        # 创建队列管理器  根据需要制定线程数量num_threads，这里为1
        qr = tf.train.QueueRunner(example_queue, [enqueue_example] * 1)
        # 将qr加入图的QueueRunner集合中
        tf.train.add_queue_runner(qr)

        # 转换label格式
        label_op = tf.cast(dequeue_example, tf.int32)

    with tf.Session(graph=g) as sess:
        # 创建线程协调器
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 出队数据
        labels = np.zeros((40000,))
        for i in range(40000):
            data = sess.run(dequeue_example)
            label = sess.run(label_op, feed_dict={label_op : data[1]})
            labels[i] = label
            if i % 10000 == 0:
                print(i) 
        labels = labels.astype(int)      
        # 清理
        coord.request_stop()
        coord.join(threads)

    return labels


