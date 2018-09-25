import tensorflow as tf

######################
#Session的使用方法
######################


martrix1=tf.constant([[3,3]])
martrix2=tf.constant([[2],[2]])

product = tf.matmul(martrix1,martrix2)   #martrix multiply np.dot(m1,m2)

##method 1
# sess=tf.Session()
# result1=sess.run(product)
# print(result1)
# sess.close()

#method 2

with tf.Session() as sess:
    result2=sess.run(product)
    print(result2)