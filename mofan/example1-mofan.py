# import tensorflow as tf
# import numpy as np

######################
#例子1
######################
# # #creat data
# # x_data=np.random.rand(100).astype(np.float32)
# # y_data=x_data*0.1+0.3
# #
# # ##create tensorflow structure start###
# # W=tf.Variable(tf.random_uniform([1],-1.0,1.0));
# # b=tf.Variable(tf.zeros([1]));
# #
# # y=W*x_data+b
# #
# # #计算误差
# # loss=tf.reduce_mean(tf.square(y-y_data))
# #
# # #创建优化器，并减少误差
# # optimizer=tf.train.GradientDescentOptimizer(0.5)
# # train=optimizer.minimize(loss)
# #
# # #前面做好神经网络的结构，现在初始化变量
# # init=tf.initialize_all_variables()
# # ##create tensorflow structure end###
# #
# # #激活神经网络 Very important
# # sess=tf.Session()
# # sess.run(init)
# #
# # for step in range(201):
# #     sess.run(train)
# #     if(step % 20 == 0):
# #         print(step,sess.run(W),sess.run(b))

import tensorflow as tf
import numpy as np

x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.3+0.1

W=tf.Variable(tf.random_uniform([1],-1.0,1.0))
b=tf.Variable(tf.zeros([1]))

y=W*x_data+b

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if(step % 20 == 0):
        print(step,sess.run(W),sess.run(b))


