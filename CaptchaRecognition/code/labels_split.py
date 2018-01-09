import numpy as np
from read_tfrecord import read_tfrecord

def labels_split(label):
    list_label = []
    label = int(label)
    label = str(label)
    label = label.zfill(4)
    for i in range(4):
        if label[i] == '0':
            list_label.append(10)
        else:
            break
    bit = i
    for j in range(4-i):
        
        list_label.append(int(label[bit]))
        bit = bit + 1
    return list_label


def labels_numpy(exp):
    labels = []
    for i in range(exp.shape[0]):
        labels.append(labels_split(exp[i][0]))
    
    labels = np.array(labels)
    return labels


# # # 函数运行
# # 训练
# train_x_1, train_y_1 = read_tfrecord('images_train_1.tfrecords', 4000)
# train_x_2, train_y_2 = read_tfrecord('images_train_2.tfrecords', 4000)
# train_x_3, train_y_3 = read_tfrecord('images_train_3.tfrecords', 4000)
# train_x_4, train_y_4 = read_tfrecord('images_train_4.tfrecords', 4000)
# train_x_5, train_y_5 = read_tfrecord('images_train_5.tfrecords', 4000)
# train_x_6, train_y_6 = read_tfrecord('images_train_6.tfrecords', 4000)
# train_x_7, train_y_7 = read_tfrecord('images_train_7.tfrecords', 4000)
# train_x_8, train_y_8 = read_tfrecord('images_train_8.tfrecords', 4000)

# # 拼接训练数据
# train_x = np.vstack((train_x_1, train_x_2, train_x_3, train_x_4, train_x_5, train_x_6, train_x_7, train_x_8))
# train_y = np.vstack((train_y_1, train_y_2, train_y_3, train_y_4, train_y_5, train_y_6, train_y_7, train_y_8))

# labels = labels_numpy(train_y)
# # for i in range(4):
# #     print(labels[:, i])
# # for i in range(labels.shape[0]):
# #     print(labels[i])

# def dense_to_one_hot(labels_dense, num_classes=11):
#     labels_dense = labels_dense.astype(np.uint8)   # 更换数据类型
#     num_labels = labels_dense.shape[0]
#     index_offset = np.arange(num_labels) * num_classes
#     labels_one_hot = np.zeros((num_labels, num_classes))
#     labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#     return labels_one_hot

# # # 将每一列单独转换为hot-code
# train_y_0 = dense_to_one_hot(labels[:, 0])
# print(train_y_0.shape)