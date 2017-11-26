# 读取tfrecord代码 , 并验证
for each in range(len(config)):
    config_path = 'mnist_' + config[each]['type'] + '.tfrecords'
    for num in range(3):
        [img, label] = read_tfrecord(config_path)
        sub = int('33' + str(3*each + 1 + num))
        print(label, end=' ')
        plt.subplot(sub)
        plt.imshow(img, cmap='gray')
        plt.show()