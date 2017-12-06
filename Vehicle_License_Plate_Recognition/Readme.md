** 数据分布**

![image](Figure_1.png)

**训练集验证集**：经多次检验，发现比例为3:1时，验证的正确率较高，即：75%、25%
**网络结构**
三个卷积层加池化层 一个连接层 一个输出层：激活函数 relu
   **优化条件**：
   * 正则化：dropout
   * 滑动平均模型：tf.train.ExponentialMovingAverage
   * 激活函数更改为：prelu

**测试：**  正确率
           召回率
