import tensorflow as tf
import numpy as np
import cPickle, gzip
import math
import sklearn.datasets as dataset
from DDNN_tf import local_site

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#Input parameters.  Change these accordingly
beta = .001
learning_rate = .001
n_input = 784
hidden_layers = [256,256,10]
n_classes = 10  # Schizo or not
site_size = 10000
training_epochs = 1000
batch_size = 100
step_size = 100
number_of_sites = 3
filename = "./output_ddnn.txt"

OUTPUT = []

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

data_test_s = mnist.test.images
labels_test_s = mnist.test.labels


data_test = data_test_s[np.argmax(labels_test_s,axis=1)!=3,:]
labels_test = labels_test_s[np.argmax(labels_test_s,axis=1)!=3,:]

#Get MNIST dataset.  Should implement a handler for other datasets


run_size = 10000
for n in range(10):

    DATA, LABELS = mnist.train.next_batch(site_size * number_of_sites, shuffle=True)
    data_1 = [0] * 3
    labels_1 = [0] * 3
    data_1[0], labels_1[0] = DATA[(np.argmax(LABELS, axis=1) == 0) + (np.argmax(LABELS, axis=1) == 1) + (
        np.argmax(LABELS, axis=1) == 2)], LABELS[
                                 (np.argmax(LABELS, axis=1) == 0) + (np.argmax(LABELS, axis=1) == 1) + (
                                     np.argmax(LABELS, axis=1) == 2)]
    data_1[1] = DATA[
        (np.argmax(LABELS, axis=1) == 7) + (np.argmax(LABELS, axis=1) == 8) + (np.argmax(LABELS, axis=1) == 9)]
    labels_1[1] = LABELS[
        (np.argmax(LABELS, axis=1) == 7) + (np.argmax(LABELS, axis=1) == 8) + (np.argmax(LABELS, axis=1) == 9)]

    data_1[2], labels_1[2] = DATA[(np.argmax(LABELS, axis=1) == 4) + (np.argmax(LABELS, axis=1) == 5) + (
        np.argmax(LABELS, axis=1) == 6)], LABELS[(np.argmax(LABELS, axis=1) == 4) + (np.argmax(LABELS, axis=1) == 5) + (
        np.argmax(LABELS, axis=1) == 6)]

    #Build the sites.  Each site should be a dictionary.
    #TODO: change 'local_site' to store the dictionary
    sites = []
    for i in range(number_of_sites):
        #TODO: handled by future handler


        sites.append(local_site('site_'+str(i),n_input,hidden_layers))
        if len(data_1[i]) < run_size:
            run_size = len(data_1[i])
        sites[i].feed(data_1[i],labels_1[i])

    sess = tf.Session()


    X,Y,cost,WB_1, grads, acc = sites[0].get()

    x_2,y_2,cost_2,WB_2, grads_2, acc_2 = sites[1].get()

    x_3,y_3,cost_3,WB_3, grads_3, acc_3 = sites[2].get()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    grads_out = []

    for j in range(number_of_sites):
        for i in range(len(grads)):
            grads_out.append(grads[i]+(grads_2[i]+ grads_3[i])/number_of_sites)

    grads_and_vars = list(zip(grads_out, WB_1+WB_2+WB_3))

    train_op = optimizer.apply_gradients(grads_and_vars)

    init = tf.global_variables_initializer()

    sess.run(init)
    output = []
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = run_size/batch_size#int(mnist.train.num_examples/batch_size)#

        for i in range(total_batch):
            s = i*batch_size
            batch_xs, batch_ys = sites[0].get_data(range(s,s+batch_size))#moons_data[i*batch_size:(i+1)*batch_size],moons_labels[i*batch_size:(i+1)*batch_size]##

            batch_xs_2, batch_ys_2 = sites[1].get_data(range(s,s+batch_size))#moons_data_2[i*batch_size:(i+1)*batch_size],moons_labels_2[i*batch_size:(i+1)*batch_size]#

            batch_xs_3, batch_ys_3 = sites[2].get_data(range(s, s + batch_size))
            # Fit training using batch data
            r = sess.run([acc], feed_dict={X: batch_xs, Y: batch_ys})
            f_pass = sess.run([train_op, cost], feed_dict={X: batch_xs, Y: batch_ys,
                                                         x_2:batch_xs_2, y_2:batch_ys_2,
                                                         x_3:batch_xs_3, y_3:batch_ys_3})

        if epoch%step_size == 0:
                r = sess.run([acc], feed_dict={X:data_test,Y:labels_test})
                #r = sess.run([acc], feed_dict={X:test_d,Y:test_l})
                output.append(r[0])

                print r

    OUTPUT.append(output)
    np.savetxt(filename, OUTPUT,delimiter=',')