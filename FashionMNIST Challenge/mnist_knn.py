from numpy import *
import operator
import os

# 分类函数

def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row
  
    # step 1:计算欧式距离
    diff = tile(newInput, (numSamples, 1)) - dataSet # 单个测试样本与训练样本中的每一个数据做比较
    squaredDiff = diff ** 2 
    squaredDist = sum(squaredDiff, axis = 1) 
    distance = squaredDist ** 0.5
  
    # step 2: 对距离进行分类
    sortedDistIndices = argsort(distance)     # 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
  
    classCount = {} 
    for i in range(k):
        # step 3: 找最小距离
        voteLabel = labels[sortedDistIndices[i]][0]
  
        # step 4: 分类与次数一一对应
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # step 5: 返回投票结果
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex


# 分类正确率
def testHandWritingClass():
    print("step 1: load data...")
    [train_x, train_y] = read_tfrecord('mnist_train.tfrecords', 55000)
    [test_x, test_y] = read_tfrecord('mnist_test.tfrecords', 500)
    
     
    print("step 2: testing...")
    numTestSamples = test_x.shape[0]
    matchCount = 0
    for i in range(numTestSamples):
        predict = kNNClassify(test_x[i], train_x, train_y, 5)
        if predict == test_y[i]:
            matchCount += 1
        if i % 100 == 0:
            print(i)
    accuracy = float(matchCount) / numTestSamples

    print("step 3: show the result...") 
    print('The classify accuracy is: %.2f%%' % (accuracy * 100))

testHandWritingClass()