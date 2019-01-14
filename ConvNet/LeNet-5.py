import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

# 训练数据
x = tf.placeholder("float", shape=[None, 784])
# 训练标签数据
y_ = tf.placeholder("float", shape=[None, 10])
# 把x更改为4维张量，第1维代表样本数量，第2维和第3维代表图像长宽， 第4维代表图像通道数, 1表示黑白
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层：卷积层
# 过滤器大小为5*5, 当前层深度为1， 过滤器的深度为32
with tf.name_scope("conv1_weights"):
    conv1_weights = tf.get_variable("conv1_weights", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
    tf.summary.histogram("conv1_weights",conv1_weights)

with tf.name_scope("conv1_biases"):
    conv1_biases = tf.get_variable("conv1_biases", [32], initializer=tf.constant_initializer(0.0))
    tf.summary.histogram("conv1_biases", conv1_biases)

with tf.name_scope("conv1"):
# 移动步长为1, 使用全0填充
    conv1 = tf.nn.conv2d(x_image, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    tf.summary.histogram("conv1",conv1)

# 激活函数Relu去线性化
with tf.name_scope("relu1"):
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    tf.summary.histogram("relu1", relu1)
# 第二层：最大池化层
# 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
with tf.name_scope("pool1"):
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    tf.summary.histogram("pool1", pool1)

# 第三层：卷积层
with tf.name_scope("conv2_weights"):
    conv2_weights = tf.get_variable("conv2_weights", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(
    stddev=0.1))  # 过滤器大小为5*5, 当前层深度为32， 过滤器的深度为64
    tf.summary.histogram("conv2_weights", conv2_weights)
with tf.name_scope("conv2_biases"):
    conv2_biases = tf.get_variable("conv2_biases", [64], initializer=tf.constant_initializer(0.0))
    tf.summary.histogram("conv2_biases", conv2_biases)
conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')  # 移动步长为1, 使用全0填充
with tf.name_scope("relu2"):
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    tf.summary.histogram("relu2", relu2)
# 第四层：最大池化层
# 池化层过滤器的大小为2*2, 移动步长为2，使用全0填充
with tf.name_scope("pool2"):
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    tf.summary.histogram("pool2", pool2)
with tf.name_scope("fc1_weights"):
# 第五层：全连接层
    fc1_weights = tf.get_variable("fc1_weights", [7 * 7 * 64, 1024],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))  # 7*7*64=3136把前一层的输出变成特征向量
    tf.summary.histogram("fc1_weights", fc1_weights)
with tf.name_scope("fc1_biases"):
    fc1_baises = tf.get_variable("fc1_baises", [1024], initializer=tf.constant_initializer(0.1))
    tf.summary.histogram("fc1_baises", fc1_baises)
pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])
with tf.name_scope("fc1"):
    fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_baises)
    tf.summary.histogram("fc1", fc1)
# 为了减少过拟合，加入Dropout层
keep_prob = tf.placeholder(tf.float32)
with tf.name_scope("fc1_dropout"):
    fc1_dropout = tf.nn.dropout(fc1, keep_prob)
    tf.summary.histogram("fc1_dropout", fc1_dropout)
# 第六层：全连接层
with tf.name_scope("fc2_weights"):
    fc2_weights = tf.get_variable("fc2_weights", [1024, 10],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))  # 神经元节点数1024, 分类节点10
    tf.summary.histogram("fc2_weights", fc2_weights)
with tf.name_scope("fc2_biases"):
    fc2_biases = tf.get_variable("fc2_biases", [10], initializer=tf.constant_initializer(0.1))
    tf.summary.histogram("fc2_biases", fc2_biases)
with tf.name_scope("fc2"):
    fc2 = tf.matmul(fc1_dropout, fc2_weights) + fc2_biases
    tf.summary.histogram("fc2", fc2)
# 第七层：输出层
# softmax
with tf.name_scope("y_conv"):
    y_conv = tf.nn.softmax(fc2)
    tf.summary.histogram("y_conv", y_conv)

# 定义交叉熵损失函数
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    tf.summary.histogram("cross_entropy", cross_entropy)
    tf.summary.scalar("loss",cross_entropy)
# 选择优化器，并让优化器最小化损失函数/收敛, 反向传播
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真实值  
# 判断预测值y和真实值y_中最大数的索引是否一致，y的值为1-10概率  
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# 用平均值来统计测试准确率  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("LeNet-5/train",sess.graph)
test_writer = tf.summary.FileWriter("LeNet-5/test",sess.graph)
# 开始训练
sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})  # 评估阶段不使用Dropout
        train_result = sess.run(merged, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})
        test_result = sess.run(merged, feed_dict={x: mnist.test.images, y_: mnist.test.labels,keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # 训练阶段使用50%的Dropout

# 在测试数据上测试准确率
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


