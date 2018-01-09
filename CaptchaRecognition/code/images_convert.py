import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob
import random
from csv_txt import labels_csv_numpy


def images_convert():
    labels = labels_csv_numpy()    # 读取标签
    rst = []
    # path
    path_initial = '../data/captcha/images/'
    path = path_initial + '/*.*'
    count = 0
    for jpgfile in glob.glob(path):
        # if count % 1000 == 0:
        #     print(count)
        count = count + 1
        img = Image.open(jpgfile)
        # resize图片
        new_img = img.resize((48, 32), Image.BILINEAR)
        # 将图片转化为灰度图
        gray = new_img.convert('L')
        # reshape 读取
        gray = np.array(gray, dtype='uint8')
        gray = gray.reshape(32, 48)
        # 截取图片中数据部分
        # add = re.findall(r"\d+\.?\d*", jpgfile)
        add = jpgfile[23: -4]
        # add = float(add[0])
        add = int(add)
        # 从列表中读取验证码
        code = labels[add]
        # # 读取
        # print(code)
        # fig = plt.figure()
        # plotwindow = fig.add_subplot(111)
        # plt.imshow(gray, cmap='gray')    
        # plt.show()
        # reshape 存储
        gray = gray.reshape(1536,)
        gray = gray.tolist()   # numpy -> list
        gray.insert(0, code)
        # 保存
        rst.append(gray)
        # if count == 1000:
        #     break
    print('data OK!')
    # 将4种验证码分开
    list_zo = []
    for length in range(1, 5):
        list_temp = []
        for each in range(len(rst)):
            code = str(rst[each][0])
            if len(code) == length:
                list_temp.append(rst[each])
        list_zo.append(list_temp)
    print('code ok!') 
                    
    # 分为10类 ：train 4000*8  validation 4000*1  test 4000*1
    list_rst = []
    for i in range(10):
        list_temp = []
        for j in range(4):
            list_temp_temp = list_zo[j]
            random.shuffle(list_temp_temp)  
            list_temp.extend(list_temp_temp[0:1000])
        # list_temp = np.array(list_temp)
        # list_temp = list_temp.reshape((-1, 2431))
        # list_temp = list_temp.tolist()   # numpy -> list;
        random.shuffle(list_temp)
        list_rst.append(list_temp)

    list_rst = np.array(list_rst)
    print('class ok!')
    
    return list_rst


# data = images_convert()
# for i in range(len(data)):
#     print(data[i].shape[0])




