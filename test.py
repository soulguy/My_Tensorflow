import tensorflow as tf
import numpy as np

def add_layer(input,input_size,output_size,n_layer,activation_function=None):
    layer_name = "layer%s" % n_layer
    with tf.name_scope("Weight"):
        Weights = tf.Variable(tf.random_normal([input_size,output_size],name='W'))
        tf.summary.histogram(layer_name+"/weight",Weights)
    with tf.name_scope("biase"):
        biases = tf.Variable(tf.zeros([1,output_size])+0.1)
        tf.summary.histogram(layer_name+"/biases",biases)
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b=tf.matmul(input,Weights)+biases

        if activation_function is None:
            output=Wx_plus_b
        else:
            output=activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+"output",output)
    return output

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5 + noise

xs=tf.placeholder(tf.float32,[None,1],name="input")
ys=tf.placeholder(tf.float32,[None,1],name="output")

layer1=add_layer(x_data,1,10,n_layer=1,activation_function=tf.nn.relu)

prediction=add_layer(layer1,10,1,n_layer=2,activation_function=None)

with tf.name_scope("loss"):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-y_data),
                                      reduction_indices=[1]))
    tf.summary.scalar("loss",loss)

with tf.name_scope("train"):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("/logs/",sess.graph)
sess.run(init)
for i in range(10000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)