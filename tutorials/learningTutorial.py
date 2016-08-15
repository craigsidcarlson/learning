#https://www.youtube.com/watch?v=dYhrCUFN0eM
import tensorflow as tf
import numpy as np
import os

#One feature (house size)
x = tf.placeholder(tf.float32, [None, 1])

#These variables are going to be trained using the gradient approach
#That is to look for where the negative change in the cost is the createst and 
#adjusting the variables in that direction

#[1,1] Output one thing house price, One input "feature" (house size)
W = tf.Variable(tf.zeros([1,1]))
W_hist = tf.histogram_summary("weights", W)

#One feature house size
b = tf.Variable(tf.zeros([1]))
b_hist = tf.histogram_summary("biases", b)

#This is the tensor model
y = tf.matmul(x, W) + b

#Need to create the cost function
#One output
yInput = tf.placeholder(tf.float32, [None, 1])

#We want to minimize the cost
cost = tf.reduce_mean(tf.square(y-yInput))
cost_sum = tf.scalar_summary("cost", cost)

#Train the model with steps in 0.00001
#If the step size is too big then you can miss the correct answer
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)


sess = tf.Session()
init = tf.initialize_all_variables()
merged = tf.merge_all_summaries()
sess.run(init)
writer = tf.train.SummaryWriter("/event_log_dir", sess.graph_def)

steps = 200


#Create fake data for y=W.x + b where W=2, b=0
for i in range(steps):
	xs = np.array([[i]])
	ys = np.array([[2*i]])

	#Train
	feed = {x: xs, yInput:ys}
	sess.run(train_step, feed_dict=feed)
	if i % 10 == 0:
		all_feed = {x:xs, yInput: ys}
		result = sess.run(merged, feed_dict=all_feed)
		#writer.add_summary(result, i)

	#print ("After %d iteration: " % i)
	#print ("W %f" % sess.run(W))
	#print ("b %f" % sess.run(b))


#Summary 
#Build the model(linear regression, logistic regression, nueral networks)
#DEfine a cost function to determine how well the data fits
#collect and prepare data
#TRain model to tweak model based on the cost function

#Tensor flow calculations are run in parallel