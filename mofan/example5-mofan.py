import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
###给定输入值，给出输出的值，通过梯度下降来降低loss，提升预测值的准确性



######################
#添加层
######################

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name="layer%s" % n_layer
    with tf.name_scope(layer_name):

        with tf.name_scope("weight"):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name="W")
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope("biase"):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram(layer_name + '/biase', biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases

        if activation_function is None:
            outputs = Wx_plus_b;
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+"/outputs",outputs)
        return outputs


#####################
#输入层一个神经元 隐藏层是个神经元 输出层一个神经元
#####################
x_data = np.linspace(-1,1,300)[:,np.newaxis] #在某一个特性（维度） -1~1 之间 300个数量
noise = np.random.normal(0,0.05,x_data.shape) #noise调整数据，使数据更像一些真实的数据 0-mean 0.05方差 格式是x_data的格式
y_data = np.square(x_data)-0.5 + noise

with tf.name_scope("input"): #整个input图层因此就包含了x_input和y_input
    xs = tf.placeholder(tf.float32,[None,1],name="x_input") ##必须要定义placeholder参数的类型 tf.float32
    ys = tf.placeholder(tf.float32,[None,1],name="y_input")

layer1 = add_layer(xs,1,10,n_layer=1,activation_function = tf.nn.relu)

#最后输出的没有使用激活函数就是线性函数的处理
prediction = add_layer(layer1,10,1,n_layer=2,activation_function = None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                    reduction_indices=[1]))
    tf.summary.scalar("loss",loss)

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

merged=tf.summary.merge_all()
wiriter = tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)

##tensorboard 可视化scalar
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        result=sess.run(merged,
                        feed_dict={xs:x_data,ys:y_data})
        wiriter.add_summary(result,i)


##可视化学习的过程
fig = plt.figure()  #创建一个用来显示图形输出的一个窗口对象

ax = fig.add_subplot(1,1,1)# add_subpot(x,y,z)  将画布分割成x行y列，图像画在从左到右从上到下的第z块

ax.scatter(x_data,y_data)

plt.ion()  #打开交互模式

plt.show() #显示图像

for i in range(1001):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})   ##只要是通过placeholder进行的运算就到使用feed_dict={}及feed相关的参数
    if(i% 20 == 0):
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'-r',lw = 5)  #画出不断调整的线
        plt.pause(0.1)
##可视化学习的过程