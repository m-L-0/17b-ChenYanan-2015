import numpy as np
import matplotlib.pyplot as plt
from images_convert import images_convert


# 函数
def data_dict(data):
    dict_rst = {}
    for i in range(data.shape[0]):
        if data[i][0] > 10:
            k = chr(data[i][0])
        else:
            k = data[i][0]
        dict_rst[k] = dict_rst.get(k, 0) + 1

    return dict_rst
# 展示
# 读取数据
# train, validation = images_convert()
# # train
# dict_train = data_dict(train)
# labels_train = tuple(dict_train.keys())
# count_train = np.array(list(dict_train.values()))
# # validation
# dict_validation = data_dict(validation)
# labels_validation = tuple(dict_validation.keys())
# count_validation = np.array(list(dict_validation.values()))

# x = np.arange(len(labels_train))   # 共用

# plt.figure()
# ax1 = plt.axes([0.16, 0.12, 0.77, 0.77])

# rects1 = ax1.bar(x, count_train, align='center', color='b', alpha=0.5, edgecolor='white')
# rects2 = ax1.bar(x, -count_validation, align='center', color='y', alpha=0.5, edgecolor='white')
# # 标数
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         if height < 0:
#             plt.text(rect.get_x(), 1.03*height, '%s' % int(-height))
#         else:
#             plt.text(rect.get_x(), 1.03*height, '%s' % int(height))
# autolabel(rects1)
# autolabel(rects2)
# plt.xlim(-1, 34)
# plt.xlabel('labes', size=14, color='k')
# plt.ylabel('count', size=14, color='k')
# ax1.set_xticks(x)
# ax1.set_xticklabels(labels_train)

# plt.show()



# 任务
# 直方图
data = images_convert('renwu')

dict_data = data_dict(data)
labels_data = tuple(dict_data.keys())
count_data = np.array(list(dict_data.values()))

x = np.arange(len(labels_data))   

plt.figure()
ax1 = plt.axes([0.16, 0.12, 0.77, 0.77])

rects = ax1.bar(x, count_data, align='center', color='g', alpha=0.5)
# 标数
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height < 0:
            plt.text(rect.get_x(), 1.03*height, '%s' % int(-height))
        else:
            plt.text(rect.get_x(), 1.03*height, '%s' % int(height))
autolabel(rects)
plt.xlim(-1, 34)
plt.xlabel('labes', size=14, color='k')
plt.ylabel('count', size=14, color='k')
ax1.set_xticks(x)
ax1.set_xticklabels(labels_data)

plt.show()


# pie
 
plt.axes(aspect = 1)#使x y轴比例相同
 
plt.pie(x=count_data, labels=labels_data, autopct='%.0f%%', pctdistance=0.7)#autopct显示百分比
plt.title('label-count')
plt.show()