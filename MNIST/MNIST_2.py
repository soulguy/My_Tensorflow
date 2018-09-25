import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

sess = tf.InteractiveSession()
'''我们通过为输入图像和目标输出类别创建节点，来开始构建计算图。'''
x = tf.placeholder('float',shape=[None, 784])

y_ = tf.placeholder('float',shape=[None, 10])

'''我们现在为模型定义权重W和偏置b。 
   可以将它们当作额外的输入量，但是TensorFlow有一个更好的处理方式：变量。
   一个变量代表着TensorFlow计算图中的一个值，能够在计算过程中使用，甚至进行修改。
   在机器学习的应用过程中，模型参数一般用Variable来表示'''

'''我们在调用tf.Variable的时候传入初始值。
   在这个例子里，我们把W和b都初始化为零向量。
   W是一个784x10的矩阵（因为我们有784个特征和10个输出值）。
   b是一个10维的向量（因为我们有10个分类）。'''
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x,W)+b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

'''每一步迭代，我们都会加载50个训练样本，然后执行一次train_step，
   并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。'''
for i in range(100):
    batch = mnist.train.next_batch[50]
    train_step.run(feed_dict={x: batch[0],y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.case(correct_prediction,'folat'))
print(accuracy.eval(feed_dict={x: mnist.test.images,y_: mnist.test.labels}))
