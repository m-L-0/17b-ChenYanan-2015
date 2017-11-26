import numpy as np
from numpy import array
import tensorflow as tf
import matplotlib.pyplot as plt
from random import choice, shuffle



# 一、实现主成分分析

def pca(data_mat, top_n_feat=99999999):
    # 获取数据条数和每条的维数
    num_data, dim = data_mat.shape

    # 数据中心化，即指变量减去它的均值
    mean_vals = data_mat.mean(axis=0)  # shape:(784,)
    mean_removed = data_mat - mean_vals  # shape:(5000, 784)

    cov_mat = np.cov(mean_removed, rowvar=0)    # 计算协方差矩阵

    # 计算特征值(Find eigenvalues and eigenvectors)
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))  # 计算特征值和特征向量

    eig_val_index = np.argsort(eig_vals)  # 对特征值进行从小到大排序，argsort返回的是索引，即下标

    eig_val_index = eig_val_index[:-(top_n_feat + 1): -1]  # 最大的前top_n_feat个特征的索引
    reg_eig_vects = eig_vects[:, eig_val_index] 

    # 将数据转到新空间
    low_d_data_mat = mean_removed * reg_eig_vects
    recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals  # 根据前几个特征向量重构回去的矩阵，shape:(5000, 784)

    return low_d_data_mat, recon_mat


# 二、实现k-means 算法

def KMeansCluster(vector, noofclusters):
    noofclusters = int(noofclusters)
    assert noofclusters < len(vector)

    # 找出每个向量的维度
    num_data, dim = vector.shape
    # 将复数矩阵转换为浮点数矩阵
    vectors = np.zeros((num_data, 2))
    for count in range(num_data):
        temp = [vector.real[count, 0], vector.real[count, 1]]
        vectors[count, :] = temp

    # 辅助随机地从可得的向量中选取中心点
    vector_indices = list(range(len(vectors)))
    shuffle(vector_indices)

    # 计算图
    # 创建一个默认的计算流的图用于整个算法中，这样就保证了当函数被多次调用时，默认的图并不会被从上一次调用时留下的未使用的OPS或者Variables挤满
    graph = tf.Graph()
    with graph.as_default():
        # 计算的会话
        sess = tf.Session()
        # 构建基本的计算的元素
        # 首先需要保证每个中心点都会存在一个Variable矩阵
        # 从现有的点集合中抽取出一部分作为默认的中心点
        centroids = [tf.Variable((vectors[vector_indices[i]]))
                     for i in range(noofclusters)]
        # 创建一个placeholder用于存放各个中心点可能的分类的情况
        centroid_value = tf.placeholder("float64", [dim])
        cent_assigns = []
        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centroid_value))
        # 对于每个独立向量的分属的类别设置为默认值0
        assignments = [tf.Variable(0) for i in range(len(vectors))]
        # 这些节点在后续的操作中会被分配到合适的值
        assignment_value = tf.placeholder("int32")
        cluster_assigns = []
        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment,
                                             assignment_value))
        # 下面创建用于计算平均值的操作节点
        # 输入的placeholder
        mean_input = tf.placeholder("float", [None, dim])
        # 节点/OP接受输入，并且计算0维度的平均值，譬如输入的向量列表
        mean_op = tf.reduce_mean(mean_input, 0)
        # 用于计算欧几里得距离的节点
        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(
            v1, v2), 2)))
        # 这个OP会决定应该将向量归属到哪个节点
        # 基于向量到中心点的欧几里得距离
        # Placeholder for input
        centroid_distances = tf.placeholder("float", [noofclusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)
        # 初始化所有的状态值
        # 这会帮助初始化图中定义的所有Variables。Variable-initializer应该定义在所有的Variables被构造之后，这样所有的Variables才会被纳入初始化
        init_op = tf.global_variables_initializer()
        # 初始化所有的变量
        sess.run(init_op)
        # 集群遍历
        # 接下来在K-Means聚类迭代中使用最大期望算法。为了简单起见，只让它执行固定的次数，而不设置一个终止条件
        noofiterations = 20
        for iteration_n in range(noofiterations):
            print('迭代次数', iteration_n)

            # 期望步骤
            # 基于上次迭代后算出的中心点的未知
            # the _expected_ centroid assignments.
            # 首先遍历所有的向量
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]
                # 计算给定向量与分配的中心节点之间的欧几里得距离
                distances = [sess.run(euclid_dist, feed_dict={
                    v1: vect, v2: sess.run(centroid)})
                             for centroid in centroids]   
                # 下面可以使用集群分配操作，将上述的距离当做输入
                assignment = sess.run(cluster_assignment, feed_dict = {
                    centroid_distances: distances})
                # 接下来为每个向量分配合适的值
                sess.run(cluster_assigns[vector_n], feed_dict={
                    assignment_value: assignment})
            # 最大化的步骤
            # 基于上述的期望步骤，计算每个新的中心点的距离从而使集群内的平方和最小
            for cluster_n in range(noofclusters):
                # 收集所有分配给该集群的向量
                assigned_vects = [vectors[i] for i in range(len(vectors))
                                  if sess.run(assignments[i]) == cluster_n]
                # 计算新的集群中心点
                new_location = sess.run(mean_op, feed_dict={
                    mean_input: array(assigned_vects)})
                # 为每个向量分配合适的中心点
                sess.run(cent_assigns[cluster_n], feed_dict={
                    centroid_value: new_location})
        # 返回中心节点和分组
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments


# 三、获取数据
[data_mat, data_label] = read_tfrecord('mnist_train.tfrecords', 5000)

# 只取最重要的两个特征
low_2_mat, recon_784_mat = pca(data_mat, 2)

# 四、获取中心点 
k = 10
center, result = KMeansCluster(low_2_mat, k)
print(center)

# 五、画出聚类结果，每一类用一种颜色
colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
n_clusters = 10
for i in range(n_clusters):
    indexs = [index for index in range(5000) if result[index] == i]
    x0 = low_2_mat[indexs, 0]
    x1 = low_2_mat[indexs, 1]
    y_i = data_label[indexs]
    for j in range(len(x0)):
        plt.text(x0[j], x1[j], str(int(y_i[j])), color=colors[i], 
                fontdict={'weight': 'bold', 'size': 9})
    plt.scatter(center[i][0], center[i][1], marker='x', color=colors[i], linewidths=12)
plt.axis([-10, 10, -10, 10])
plt.show()