import tensorflow as tf


######################
#Variable介绍
######################

state = tf.Variable(0,name='counter')
print(state.name)

#tf.constant(1) 定义一个等于1的常量
one=tf.constant(1)

new_value=tf.add(state,one)

#把new_value的值加载到state上
update = tf.assign(state,new_value)

init = tf.initialize_all_variables() #must have if define variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range (3):
        sess.run(update)
        print(sess.run(state))
