import input_data
import tensorflow.examples.tutorials.mnist.input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

'''设置变量，定义模型'''

x=tf.placeholder(tf.float32,[None,784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

'''一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）'''

y_ = tf.placeholder("float", [None,10]) #即实际的分步,添加一个新的占位符用于输入正确值

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # 计算交叉熵,即成本函数,这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()#添加一个操作来初始化我们创建的变量
'''现在我们可以在一个Session里面启动我们的模型，并且初始化变量'''
sess = tf.Session()
sess.run(init)
'''然后开始训练模型，这里我们让模型循环训练1000次！
   该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，
   然后我们用这些数据点作为参数替换之前的占位符来运行train_step。'''
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

'''检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。'''
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
'''为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，
    然后取平均值。例如，[True, False, True, True] 会变成 [1,0,1,1] ，
    取平均值后得到 0.75.'''
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

'''最后，我们计算所学习到的模型在测试数据集上面的正确率。'''
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
