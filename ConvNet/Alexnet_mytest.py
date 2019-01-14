import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

# 定义相关参数
n_classes = 10 # 输出的类别
inputsize = 784 # 输入的维度 图片大小为[28,28]
batch_size =128
dropout = 0.75 # Dropout的概率，输出的可能性
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

x = tf.placeholder(tf.float32,[None,inputsize])
y = tf.placeholder(tf.float32,[None,n_classes])
# AlexNet 包含了五层卷积层
# 定义卷积层：
def conv2d(name, x, W, b, strides=1):
    # sconv2d wrapper,bias and relu activation
    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding='SAME')
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x,name=name)

# 定义池化层
def pooling(name, x, k=2):
    # Maxpooling wrapper
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME',name=name)

# 规范化操作
def normal(name, l_input, lsize=4):
    return tf.nn.lrn(l_input,lsize,bias=1.0,alpha=0.001/9.0,beta=0.75,name=name)

# 定义所有的网络参数

weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wf1': tf.Variable(tf.random_normal([4*4*256,  4096])),
    'wf2': tf.Variable(tf.random_normal([4096, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

bias = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bf1': tf.Variable(tf.random_normal([4096])),
    'bf2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def alex_net(x,weight,bias,dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # 第一层卷积
    conv1 = conv2d('conv1',x,weights['wc1'], bias['bc1'])
    # 下采样
    pool1 = pooling('pool1',conv1 ,k=2)
    # 规范化
    norm1 = normal('norm1',pool1, lsize=4)

    # 第二层卷积
    conv2 = conv2d('conv2',norm1,weights['wc2'], bias['bc2'])
    # 下采样
    pool2 = pooling('pool2',conv2 ,k=2)
    # 规范化
    norm2 = normal('norm2',pool2, lsize=4)

    # 第三层卷积
    conv3 = conv2d('conv3',norm2,weights['wc3'], bias['bc3'])
    # 规范化
    norm3 = normal('norm3',conv3, lsize=4)

    # 第四层卷积
    conv4 = conv2d('conv4', norm3, weights['wc4'], bias['bc4'])

    # 第五层卷积
    conv5 = conv2d('conv5', conv4, weights['wc5'], bias['bc5'])
    # 下采样
    pool5 = pooling('pool5',conv5, k=2)
    # 规范化
    norm5 = normal('norm5',pool5, lsize=4)

    # 全连接层1
    fc1 = tf.reshape(norm5, [-1, weights['wf1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wf1']), bias['bf1'])
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # 全连接层2
    fc2 = tf.reshape(fc1, [-1, weights['wf2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wf2']), bias['bf2'])
    fc2 = tf.nn.relu(fc2)
    # dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # 输出层
    out = tf.add(tf.matmul(fc2, weights['out']), bias['out'])
    return out


# 构建模型，定义损失函数和优化器，并构建评估函数
pred = alex_net(x,weights,bias,keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimization = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# 评估函数
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

# 初始化 变量
init = tf.global_variables_initializer()

# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimization,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
        if i%50 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print("step ",i,", Minibatch Loss= ",loss,", Training Accuracy= ",acc)
    print("Optimazer finishing!")
    # 计算测试集的精度
    test_accuracy = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.})
    print("test accuracy is"+test_accuracy)