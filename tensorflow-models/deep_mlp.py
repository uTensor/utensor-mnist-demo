# This script is based on:
# https://www.tensorflow.org/get_started/mnist/pros

import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util as gu

FLAGS = None

def deepnn(x):
  with tf.name_scope("Layer1"):
    W_fc1 = weight_variable([784, 128], name='W_fc1')
    b_fc1 = bias_variable([128], name='b_fc1')
    a_fc1 = tf.add(tf.matmul(x, W_fc1), b_fc1, name="zscore")
    h_fc1 = tf.nn.relu(a_fc1)

  with tf.name_scope("Layer2"):
    W_fc2 = weight_variable([128, 64], name='W_fc2')
    b_fc2 = bias_variable([64], name='b_fc2')
    a_fc2 = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2, name="zscore")
    h_fc2 = tf.nn.relu(a_fc2)
  
  with tf.name_scope("OuputLayer"):
    W_fc3 = weight_variable([64, 10], name='W_fc3')
    b_fc3 = bias_variable([10], name='b_fc3')
    y_pred = tf.add(tf.matmul(h_fc2, W_fc3), b_fc3, name="prediction")

  return y_pred


def weight_variable(shape, name):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name)


def bias_variable(shape, name):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name)


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784], name="x")

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10], name="y")

  # Build the graph for the deep net
  y_pred = deepnn(x)

  with tf.name_scope("Loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, 
                                                            logits=y_pred)
    loss = tf.reduce_mean(cross_entropy, name="cross_entropy_loss")
  train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, name="train_step")
  
  with tf.name_scope("Prediction"): 
    correct_prediction = tf.equal(tf.argmax(y_pred, 1, name='y_pred'), 
                                  tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels}))
    saver.save(sess, "./my-model/model.ckpt")
    out_nodes = [y_pred.op.name, y_.op.name, cross_entropy.op.name,
                 correct_prediction.op.name, accuracy.op.name]
    sub_graph_def = gu.convert_variables_to_constants(sess, sess.graph_def, out_nodes)
    graph_path = tf.train.write_graph(sub_graph_def, 
                                      "./my-model", "train.pb", 
                                      as_text=False)
    print('written graph to: %s' % graph_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
