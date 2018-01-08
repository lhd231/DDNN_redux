import tensorflow as tf
import numpy as np
import cPickle, gzip
import math
import sklearn.datasets as dataset
from DDNN_tf import local_site

from tensorflow.examples.tutorials.mnist import input_data


def make_one_hot(target,labels):
    targets = np.zeros((len(target),labels))
    targets[np.arange(len(target)),target - 1] = 1
    return targets

beta = .001
learning_rate = .001
n_input = 784
hidden_layers = [256,256,10]
n_classes = 10  # Schizo or not
site_size = 30000
training_epochs = 1000
batch_size = 300
step_size = 100
filename = "./output_sing.txt"

OUTPUT = []

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

data_test_s = mnist.test.images
labels_test_s = mnist.test.labels


data_test = data_test_s[np.argmax(labels_test_s,axis=1)!=3,:]
labels_test = labels_test_s[np.argmax(labels_test_s,axis=1)!=3,:]
for n in range(10):
    #moons_data,moons_labels = dataset.make_moons(400,shuffle=True)
    #moons_data_2,moons_labels_2 = dataset.make_moons(400,shuffle=True)

    #test_data, test_labels = dataset.make_moons(100,shuffle=True)

    #test_labels = make_one_hot(test_labels,n_classes)
    #moons_labels = make_one_hot(moons_labels, n_classes)
    #moons_labels_2 = make_one_hot(moons_labels_2, n_classes)
    def multilayer_perceptron(x, weights, biases):
        layer = x
        for w,b in zip(weights, biases):
            layer = tf.nn.relu(tf.add(tf.matmul(layer,w),b))

        return layer


    DATA, LABELS = mnist.train.next_batch(site_size, shuffle=True)

    data_1, labels_1 = DATA[(np.argmax(LABELS,axis=1)==0) + (np.argmax(LABELS,axis=1)==1) + (np.argmax(LABELS,axis=1)==2)], LABELS[(np.argmax(LABELS,axis=1)==0) + (np.argmax(LABELS,axis=1)==1) + (np.argmax(LABELS,axis=1)==2)]
    data_1 = np.concatenate((data_1, DATA[(np.argmax(LABELS,axis=1)==7) + (np.argmax(LABELS,axis=1)==8) + (np.argmax(LABELS, axis=1)==9)]), axis=0)
    labels_1 = np.concatenate((labels_1,LABELS[(np.argmax(LABELS,axis=1)==7) + (np.argmax(LABELS,axis=1)==8) + (np.argmax(LABELS,axis=1)==9)]),axis=0)
    data_1 = np.concatenate((data_1,DATA[(np.argmax(LABELS, axis=1)==4) + (np.argmax(LABELS, axis=1)==5) + (np.argmax(LABELS, axis=1)==6)]),axis=0)
    labels_1 = np.concatenate((labels_1, LABELS[(np.argmax(LABELS, axis=1)==4) + (np.argmax(LABELS, axis=1)==5) + (np.argmax(LABELS, axis=1)==6)]),axis=0)



    sess = tf.Session()
    l = local_site('site1',n_input,hidden_layers)
    l.feed(data_1,labels_1)
    X,Y,cost,WB_1, grads, acc = l.get()



    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    print len(tf.trainable_variables())
    #grads = tf.gradients(xs=tf.trainable_variables()[:6], ys=cost)
    #grads = grads + tf.gradients(xs=tf.trainable_variables()[6:], ys=cost_2)

    #grads = tf.gradients(xs = tf.trainable_variables(),ys = cost)
    for x in tf.trainable_variables():
        print x.name
    for x in grads:
        print type(x)
        if x != None:
            print x.name
    grads_out = grads


    grads_and_vars = list(zip(grads_out, WB_1))

    train_op = optimizer.apply_gradients(grads_and_vars)


    init = tf.global_variables_initializer()

    sess.run(init)
    output = []
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = site_size/batch_size#int(mnist.train.num_examples/batch_size)#

        for i in range(total_batch):
            s = i*batch_size
            batch_xs, batch_ys = data_1[s:s+batch_size], labels_1[s:s+batch_size]#moons_data[i*batch_size:(i+1)*batch_size],moons_labels[i*batch_size:(i+1)*batch_size]##


            # Fit training using batch data
            f_pass = sess.run([train_op, cost], feed_dict={X: batch_xs,
                                                       Y: batch_ys})

        if epoch%step_size == 0:
                r = sess.run([acc], feed_dict={X:data_test,Y:labels_test})
                #r = sess.run([acc], feed_dict={X:test_d,Y:test_l})
                output.append(r[0])
                print "next"
                print epoch
                print r
                #np.savetxt(filename, output, delimiter=' ')
    OUTPUT.append(output)
    np.savetxt(filename, OUTPUT, delimiter=',')