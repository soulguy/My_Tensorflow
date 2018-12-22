from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

sess = tf.InteractiveSession()

# 定义输入 图片规格为28
# 训练数据
xs = tf.placeholder("float", shape=[None, 784])
# 训练标签数据
ys = tf.placeholder("float", shape=[None, 10])

input_images = tf.reshape(xs, [-1, 28, 28, 1])

# 第一层：卷积层C1

# 共享权值
with tf.name_scope('conv1_weights'):
    conv1_weights = tf.get_variable('conv1_weights', [5, 5, 1, 32],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    tf.summary.histogram('conv1_weights', conv1_weights)
# 每个滤波器扫描完后加上的偏移量
with tf.name_scope('conv1_biases'):
    conv1_biases = tf.get_variable('conv1_biases', [32], initializer=tf.constant_initializer(0.0))
    tf.summary.histogram('conv1_biases', conv1_biases)
# 卷积
with tf.name_scope('conv1'):
    conv1 = tf.nn.conv2d(input_images, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    tf.summary.histogram('conv1', conv1)
# 卷积完后对结果非线性转换
with tf.name_scope('conv1_relu'):
    conv1_relu = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    tf.summary.histogram('conv1_relu', conv1_relu)

# 第二层：下采样层 S2
with tf.name_scope('pooling2'):
    pooling2 = tf.nn.max_pool(conv1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    tf.summary.histogram('pooling2',pooling2)

# 第三层：卷积层C3
with tf.name_scope('conv3_weights'):
    conv3_weights = tf.get_variable('conv3_weights',[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
    tf.summary.histogram('conv3_weights',conv3_weights)
with tf.name_scope('conv3_biases'):
    conv3_biases = tf.get_variable('conv3_biases',[64],initializer=tf.constant_initializer(0.0))
    tf.summary.histogram('conv3_biases',conv3_biases)
with tf.name_scope('conv3'):
    conv3 = tf.nn.conv2d(pooling2,conv3_weights,strides=[1,1,1,1],padding='SAME')
    tf.summary.histogram('conv3',conv3)
with tf.name_scope('conv3_relu'):
    conv3_relu = tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))
    tf.summary.histogram('conv3_relu',conv3_relu)

# 第四层：下采样层 S4
with tf.name_scope('pooling4'):
    pooling4 = tf.nn.max_pool(conv3_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    tf.summary.histogram('pooling4',pooling4)

# 第五层：全连接层
with tf.name_scope('fc1_weights'):
    fc1_weights = tf.get_variable('fc1_weights', [7 * 7 * 64, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
    tf.summary.histogram('fc1_weights', fc1_weights)
with tf.name_scope('fc1_biases'):
    fc1_biases = tf.get_variable('fc1_biases', [512],initializer=tf.constant_initializer(0.1))
    tf.summary.histogram('fc1_biases', fc1_biases)
pool2_vector = tf.reshape(pooling4, [-1, 7 * 7 * 64])
with tf.name_scope('fc1'):
    fc1 = tf.nn.relu(tf.matmul(pool2_vector, fc1_weights) + fc1_biases)
    tf.summary.histogram('fc1', fc1)

# 第六层：全连接层
with tf.name_scope('fc2_weights'):
    fc2_weights = tf.get_variable('fc2_weights',[512,10],initializer=tf.truncated_normal_initializer(stddev=0.1))
    tf.summary.histogram('fc2_weights',fc2_weights)
with tf.name_scope('fc2_biases'):
    fc2_biases = tf.get_variable('fc2_biases',[10],initializer=tf.constant_initializer(0.1))
    tf.summary.histogram('fc2_biases',fc2_biases)
with tf.name_scope('fc2'):
    fc2 = tf.nn.relu(tf.matmul(fc1,fc2_weights)+fc2_biases)
    tf.summary.histogram('fc2',fc2)

# 第七层：输出层
with tf.name_scope('y_conv'):
    y_conv = tf.nn.softmax(fc2)
    tf.summary.histogram('y_conv',y_conv)

#交叉熵 计算损失函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(y_conv),reduction_indices=[1]))
    tf.summary.histogram('cross_entropy',cross_entropy)
    optimization = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correction_predicion = tf.equal(tf.argmax(ys,1),tf.argmax(y_conv,1))
accuracy = tf.reduce_mean(tf.cast(correction_predicion,tf.float32))
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("My_LeNet5/train",sess.graph)
test_writer = tf.summary.FileWriter("My_LeNet5/test",sess.graph)
sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = mnist.train.next_batch(100)
    if i%100 ==0:
        train_accuracy = accuracy.eval(feed_dict={xs:batch[0],ys:batch[1]})
        train_result = sess.run(merged,feed_dict={xs:batch[0],ys:batch[1]})
        test_result = sess.run(merged,feed_dict={xs:mnist.test.images,ys:mnist.test.labels})
        print('step ',i, ' the accuracy is : ',train_accuracy)
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
    sess.run(optimization,feed_dict={xs:batch[0],ys:batch[1]})

print("test accuracy %g" % accuracy.eval(feed_dict={xs: mnist.test.images, ys: mnist.test.labels}))