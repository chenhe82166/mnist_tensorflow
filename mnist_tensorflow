# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)#mnist.train(.images,.labels),mnist.validation(.images,.labels),mnist.test(.images,.labels)
															#一个one_hot向量除了某一位的数字是1以外其余各维度数字都是0
x=tf.placeholder(tf.float32,[None,784])
x_image=tf.reshape(x,[-1,28,28,1])
y_=tf.placeholder(tf.float32,[None,10])

W_conv1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))
h_conv1=tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))
h_conv2=tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#全连接层输入数据应为2维
W_fc1=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b_fc1=tf.Variable(tf.constant(0.1,shape=[1024]))
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#最后一层为全连接层加上softmax输出
W_fc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc2=tf.Variable(tf.constant(0.1,shape=[10]))
y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2
#损失函数
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
#定义优化器
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#准确率
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
correct_prediction=tf.cast(correct_prediction,tf.float32)   #tf.cast为类型转换函数 
                                                            #a is [1.8, 2.2], dtype=tf.float  tf.cast(a, tf.int32) ==> [1, 2] dtype=tf.int32  
accuracy=tf.reduce_mean(correct_prediction)

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(20000):
	batch_x,batch_y=mnist.train.next_batch(50)
	if i%100 ==0:
		train_accuracy=accuracy.eval(feed_dict={x:batch_x,y_:batch_y,keep_prob:1.0})
		print('step %d,training accuracy is %g' % (i,train_accuracy))
	train_step.run(feed_dict={x:batch_x,y_:batch_y,keep_prob:0.5})
print('test accuracy is %g' % accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))

