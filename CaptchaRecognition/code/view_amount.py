from csv_txt import labels_csv_numpy
import matplotlib.pyplot as plt
import numpy as np

# 导入数据
labels = labels_csv_numpy()
# 将数据转换为string
labels = labels.astype(str)
# 字典
dict_data = {}
for i in range(len(labels)):
    length = len(labels[i])
    dict_key = str(length) + 'bit verification code'
    if dict_data.get(dict_key) == None:
        dict_data[dict_key] = dict_data.get(dict_key, 0) + 1
    else:
        dict_data[dict_key] = dict_data.get(dict_key, 0) + 1

# print(dict_rst)

# 直方图
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
        plt.text(rect.get_x()+0.2*rect.get_width(), 1.03*height, '%s' % int(height))
autolabel(rects)
plt.xlim(-1, 4)
plt.xlabel('labels', size=14, color='k')
plt.ylabel('count', size=14, color='k')
ax1.set_xticks(x)
ax1.set_xticklabels(labels_data)

plt.show()


# pie
 
plt.axes(aspect = 1)#使x y轴比例相同
 
plt.pie(x=count_data, labels=labels_data, autopct='%.0f%%', pctdistance=0.7)#autopct显示百分比
plt.title('label-count')
plt.show()