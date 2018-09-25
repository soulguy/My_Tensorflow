import tensorflow as tf

######################
#placehoder介绍
######################


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output=tf.multiply(input1,input2)


#the feed_dict is like that put the input_1 and input_2 as dictionary
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
