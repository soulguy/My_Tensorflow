import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#获取minist数据集
minist = input_data.read_data_sets("MNIST_data",one_hot=True)

######################
#添加层
######################

def add_layer(inputs,in_size,out_size,activation_function=None):
        with tf.name_scope("weight"):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name="W")
            tf.summary.histogram('/weights',Weights)
        with tf.name_scope("biase"):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram('/biase', biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases

        if activation_function is None:
            outputs = Wx_plus_b;
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram("/outputs",outputs)
        return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    #每train一次就用test.minist的v_xs来生成预测值和标准预测值作比较 
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accruacy  = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accruacy,feed_dict={xs:v_xs,ys:v_ys})
    return result

xs=tf.placeholder(tf.float32,[None,784]) #28X28
ys=tf.placeholder(tf.float32,[None,10])

#add output layer
prediction =add_layer(xs,784,10,activation_function=tf.nn.softmax)
cross_entropy= tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1])) #loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    batch_xs,batch_ys=minist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 ==0:
        print("step",i,compute_accuracy(minist.test.images,minist.test.labels))



