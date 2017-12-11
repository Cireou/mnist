import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# The MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_size = 100

# Create the model, 28x28 MNIST data to 784 nodes
data = tf.placeholder(tf.float32, [None, 784])
weight = tf.Variable(tf.zeros([500, 10]))
bias = tf.Variable(tf.zeros([10]))
hl_weight = tf.Variable(tf.random_normal([784, 500]))
hl_bias = tf.Variable(tf.random_normal([500]))

# Create a hidden layer with 500 nodes, pass through relu
hidden_layer = tf.nn.relu(tf.add(tf.matmul(data, hl_weight), hl_bias))
# Create hypothesis, run through softmax 
hypothesis = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, weight),bias))

# Define cost function and optimizer
y = tf.placeholder(tf.float32, [None, 10])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
# Default learning rate is 0.001
optimizer = tf.train.AdamOptimizer().minimize(cost)

saver = tf.train.Saver()
epochs_no = 10

 
# Runs the Neural Network
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
  
    # Training
    for epoch in range(epochs_no):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples/batch_size)):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, loss = sess.run([optimizer, cost], feed_dict = {data: batch_x, y: batch_y})
            # Optimizes the weight & bias
            epoch_loss += loss
        print('Epoch', epoch, 'completed out of', epochs_no, 'loss:', epoch_loss)
    
    # Calculates the accuracy 
    correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:', accuracy.eval({data: mnist.test.images, y: mnist.test.labels}))
    
    # Save the model
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model.ckpt")
    print ("Model saved as: ", save_path)
