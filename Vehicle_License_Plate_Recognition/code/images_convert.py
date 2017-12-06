import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import random

def images_convert(use='tfrecord'):
    rst = []
    # 数字
    # path
    path_initial = '../车牌字符识别训练数据/数字/'
    for i in range(10):
        path_mi = str(i)
        path = path_initial + path_mi + '/*.*'
        for jpgfile in glob.glob(path):
            img = Image.open(jpgfile)
            # resize图片
            new_img = img.resize((24, 48), Image.BILINEAR)
            # 将图片转化为灰度图
            gray = new_img.convert('L')
            # reshape 读取
            gray = np.array(gray, dtype='uint8')
            gray = gray.reshape(48, 24)
            # # 读取
            # fig = plt.figure()
            # plotwindow = fig.add_subplot(111)
            # plt.imshow(gray, cmap='gray')    
            # plt.show()
            # reshape 存储
            gray = gray.reshape(1152,)
            gray = gray.tolist()   # numpy -> list
            gray.insert(0, i)
            # 保存
            rst.append(gray)
    print('num OK!')
    # 字母
    path_initial = '../车牌字符识别训练数据/字母/'
    for i in range(26):
        path_mi = chr(65+i)
        path = path_initial + path_mi + '/*.*'
        for jpgfile in glob.glob(path):
            img = Image.open(jpgfile)
            # resize图片
            new_img = img.resize((24, 48), Image.BILINEAR)
            # 将图片转化为灰度图
            gray = new_img.convert('L')
            # reshape 读取
            gray = np.array(gray, dtype='uint8')
            gray = gray.reshape(48, 24)
            # # 读取
            # fig = plt.figure()
            # plotwindow = fig.add_subplot(111)
            # plt.imshow(gray, cmap='gray')    
            # plt.show()
            # reshape 存储
            gray = gray.reshape(1152,)
            gray = gray.tolist()   # numpy -> list
            gray.insert(0, 65+i)
            # 保存
            rst.append(gray)
    print('letter OK!')
    random.shuffle(rst)
    rst = np.array(rst)
    np.reshape(rst, (-1, 1153))
    rst = rst.astype('uint8')
    # 选择  输出
    if use == 'tfrecord':
        return rst[:10856], rst[10856:]
    else:
        return rst


# data = images_convert()
# for i in range(len(data)):
#     print(data[i].shape[0])