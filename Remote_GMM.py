import tensorflow as tf
import numpy as np
import cPickle, gzip
import math
from sklearn.mixture import GaussianMixture
from DDNN_tf import local_site
import warnings

from tensorflow.examples.tutorials.mnist import input_data

def is_outlier(value, mean):
    """Check if value is an outlier
    """
    variance = mean - value

    return value <= lower or value >= upper

def make_one_hot(target,labels):
    targets = np.zeros((len(target),labels))
    targets[np.arange(len(target)),target - 1] = 1
    return targets

beta = .001
learning_rate = .001
n_input = 784
hidden_layers = [256,256,10]
n_classes = 10  # Schizo or not
site_size = 10000
training_epochs = 1000
batch_size = 300
step_size = 100
filename = "./output_sing_gmm_test.txt"

OUTPUT = []

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

data_test_s = mnist.test.images
labels_test_s = mnist.test.labels

print data_test_s.shape
data_test = data_test_s[np.argmax(labels_test_s,axis=1)!=3,:]

labels_test = labels_test_s[np.argmax(labels_test_s,axis=1)!=3,:]
#moons_data,moons_labels = dataset.make_moons(400,shuffle=True)
#moons_data_2,moons_labels_2 = dataset.make_moons(400,shuffle=True)

#test_data, test_labels = dataset.make_moons(100,shuffle=True)

#test_labels = make_one_hot(test_labels,n_classes)
#moons_labels = make_one_hot(moons_labels, n_classes)
#moons_labels_2 = make_one_hot(moons_labels_2, n_classes)

for n in range(10):
    print str(n)+"th iteration"
    DATA, LABELS = mnist.train.next_batch(site_size*3, shuffle=True)
    data_1, labels_1 = DATA[(np.argmax(LABELS,axis=1)==0) + (np.argmax(LABELS,axis=1)==1) + (np.argmax(LABELS,axis=1)==2)], LABELS[(np.argmax(LABELS,axis=1)==0) + (np.argmax(LABELS,axis=1)==1) + (np.argmax(LABELS,axis=1)==2)]
    data_2 = DATA[(np.argmax(LABELS,axis=1)==7) + (np.argmax(LABELS,axis=1)==8) + (np.argmax(LABELS, axis=1)==9)]
    labels_2 = LABELS[(np.argmax(LABELS,axis=1)==7) + (np.argmax(LABELS,axis=1)==8) + (np.argmax(LABELS,axis=1)==9)]

    data_3, labels_3 = DATA[(np.argmax(LABELS, axis=1)==4) + (np.argmax(LABELS, axis=1)==5) + (np.argmax(LABELS, axis=1)==6)], LABELS[(np.argmax(LABELS, axis=1)==4) + (np.argmax(LABELS, axis=1)==5) + (np.argmax(LABELS, axis=1)==6)]


    for i in range(4, 7):
        indicies = np.where(np.argmax(labels_3,axis=1) == i)[0]

        gmm = GaussianMixture(n_components=2).fit(data_3[indicies])


        num = 0
        d_fin = []
        l_fin = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x,y in zip(data_3[indicies],labels_3[indicies]):
                o = gmm.predict(x)
                if o[0].item() is  0:
                    d_fin.append(x)
                    l_fin.append(y)
                    num = num +1
        print num
        gmm = GaussianMixture(n_components=1).fit(d_fin)
        d2, _ = gmm.sample(len(d_fin))
        l2 = l_fin
        data_1 = np.concatenate((data_1, d2), axis=0)
        labels_1 =np.concatenate((labels_1, l2), axis=0)
    print "last data set"
    for i in range(7,n_classes):
        indicies = np.where(np.argmax(labels_2,axis=1) == i)[0]

        gmm = GaussianMixture(n_components=1).fit(data_2[indicies])


        num = 0
        d_fin = []
        l_fin = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x,y in zip(data_2[indicies],labels_2[indicies]):
                o = gmm.predict(x)
                if o[0].item() is  0:
                    d_fin.append(x)
                    l_fin.append(y)
                    num = num +1
        print num
        gmm = GaussianMixture(n_components=1).fit(d_fin)
        d2, _ = gmm.sample(len(d_fin))
        l2 = l_fin
        data_1 = np.concatenate((data_1, d2), axis=0)
        labels_1 =np.concatenate((labels_1, l2), axis=0)

    sess = tf.Session()

    P = np.random.permutation(len(data_1))

    print data_1.shape
    print labels_1.shape

    data_1 = data_1[P]
    labels_1 = labels_1[P]

    l = local_site('site1',n_input,hidden_layers)
    l.feed(data_1,labels_1)
    X,Y,cost,WB_1, grads, acc = l.get()


    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #grads = tf.gradients(xs=tf.trainable_variables()[:6], ys=cost)
    #grads = grads + tf.gradients(xs=tf.trainable_variables()[6:], ys=cost_2)

    #grads = tf.gradients(xs = tf.trainable_variables(),ys = cost)

    grads_out = grads


    grads_and_vars = list(zip(grads_out, WB_1))

    train_op = optimizer.apply_gradients(grads_and_vars)


    init = tf.global_variables_initializer()

    print "running"
    sess.run(init)
    output = []
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = len(labels_1)/batch_size#int(mnist.train.num_examples/batch_size)#

        for i in range(total_batch):
            s = i*batch_size
            batch_xs, batch_ys = data_1[s:s+batch_size], labels_1[s:s+batch_size]#moons_data[i*batch_size:(i+1)*batch_size],moons_labels[i*batch_size:(i+1)*batch_size]##


            # Fit training using batch data
            f_pass = sess.run([train_op, cost], feed_dict={X: batch_xs,
                                                       Y: batch_ys})

        if epoch%step_size == 0:
                print data_test.shape
                r = sess.run([acc], feed_dict={X:data_test,Y:labels_test})
                output.append(r[0])
                print r

    OUTPUT.append(output)
    np.savetxt(filename, OUTPUT, delimiter=',')