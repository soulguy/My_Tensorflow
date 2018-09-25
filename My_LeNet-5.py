# my first neural network built by myself
#LeNet-5 is first convolution neural network does obviously imporvment in recently computer vision work

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

def weight_variable(shape):
    #shape：一维整数张量或Python数组。输出张量的形状。

    #mean：类型的0 - D

    #Tensor或Python值dtype。截断正态分布的均值。

    #stddev：类型的0 - D

    #Tensor或Python值dtype。截断前的正态分布的标准偏差。

    #dtype：输出的类型。

    #seed：一个Python整数。用于为分发创建随机种子。看看tf.set_random_seed行为。

    #name：操作的名称（可选）。
    initial = tf.truncated_normal(shape,stddev=0.1)
    return initial

def bias_variable(shape):
    #value：输出类型的常量值（或列表）dtype。

    #dtype：结果张量元素的类型。

    #shape：产生张量的可选尺寸。

    #name：张量的可选名称。

    #verify_shape：布尔值，用于验证值的形状。
    initial = tf.constant(0.1,shape=shape)
    return initial

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def con2d(x,W):  #步长为1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="VALID")

def max_pool_2x2(x): #2x2 步长为2
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#第一层卷积 5x5
w_conv1 = weight_variable([5,5,1,6])
b_conv1 = bias_variable([6])
conv1 = tf.nn.relu(con2d(x_image,w_conv1)+b_conv1)
conv1_pool1 = max_pool_2x2(conv1)

#第二层卷积
w_conv2 = weight_variable([5,5,6,16])
b_conv2 = bias_variable([16])
conv2 = tf.nn.relu(con2d(conv1_pool1,w_conv2)+b_conv2)
conv2_pool2 = max_pool_2x2(conv2)

#第一层全连接

w_fc1 = weight_variable([400,120])
b_fc1 = bias_variable([120])
fc1 = tf.nn.relu(tf.matmul(conv2_pool2,w_fc1)+b_fc1)
fc1 = tf.nn.dropout(fc1,0.5)

#第二层全连接

w_fc2 = weight_variable([120,84])
b_fc2 = bias_variable([84])
fc2 = tf.nn.relu(tf.matmul(fc1,w_fc2)+b_fc2)
fc2 = tf.nn.dropout(fc2,0.5)

#第三层全连接

w_fc3 = weight_variable([84,10])
b_fc3 = bias_variable([10])
prediction = tf.matmul(fc2,w_fc3)+b_fc3

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                            reduction_indices=[1]))
train_step = tf.train.AdadeltaOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%20 ==0:
        print("step",i,compute_accuracy(mnist.test.images,mnist.test.labels))



