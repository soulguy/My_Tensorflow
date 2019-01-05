from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

xs = tf.placeholder(tf.float32,[None,784])/255.
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob: 1})
    correct_accuracy = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_accuracy,tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")

def max_pool_2x2(shape):
    return tf.nn.max_pool(shape,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
conv1_pool = max_pool_2x2(conv1)

w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
conv2 = tf.nn.relu(conv2d(conv1_pool,w_conv2)+b_conv2)
conv2_pool = max_pool_2x2(conv2)

w_flc1 = weight_variable([7*7*64,1024])
b_flc1 = bias_variable([1024])
flc1_to_flat = tf.reshape(conv2_pool,[-1,7*7*64])
flc1 = tf.nn.relu(tf.matmul(flc1_to_flat,w_flc1)+b_flc1)
flc1_drop = tf.nn.dropout(flc1,keep_prob)

w_flc2 = weight_variable([1024,10])
b_flc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(flc1_drop,w_flc2)+b_flc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob: 0.5})
    if i%50 == 0:
        print("loading...")
        print(compute_accuracy(mnist.test.images[:1000],mnist.test.labels[:1000]))
