import tensorflow as tf
import numpy as np
import cPickle, gzip
import math
from sklearn.mixture import GaussianMixture
from DDNN_tf import local_site
import warnings


def is_outlier(value, mean):
    """Check if value is an outlier
    """
    variance = mean - value

    return value <= lower or value >= upper

def make_one_hot(target,labels):
    print type(target)
    print target.shape
    targets = np.zeros((len(target),labels))
    targets[np.arange(len(target)),target - 1] = 1
    return targets

beta = .001
learning_rate = .001
n_input = 741
hidden_layers = [400,200,4]
n_classes = 4  # Schizo or not
site_size = 10000
training_epochs = 1000
batch_size = 10
step_size = 100
filename = "./output_gmm_smokes.txt"

OUTPUT = []



#data_test_s = mnist.test.images
#labels_test_s = mnist.test.labels


#moons_data,moons_labels = dataset.make_moons(400,shuffle=True)
#moons_data_2,moons_labels_2 = dataset.make_moons(400,shuffle=True)

#test_data, test_labels = dataset.make_moons(100,shuffle=True)

#test_labels = make_one_hot(test_labels,n_classes)
#moons_labels = make_one_hot(moons_labels, n_classes)
#moons_labels_2 = make_one_hot(moons_labels_2, n_classes)

IN = np.loadtxt("./Data_For_Noah.csv",delimiter=',',dtype = np.float32)
DATA = IN[:,1:]
LABELS = make_one_hot(IN[:,0].astype(int),4)
print DATA.shape
perm = np.random.permutation(len(DATA))
DATA = DATA[perm]
LABELS = LABELS[perm]
for n in range(10):
    print str(n)+"th iteration"
    d_p, l_p = DATA[:150],LABELS[:150]

    indicies_0 = np.where(np.argmax(l_p,axis=1) == 0)[0]
    indicies_1 = np.where(np.argmax(l_p, axis=1) == 1)[0]

    data_1 = d_p[indicies_0]
    data_1= np.concatenate((data_1, d_p[indicies_1]), axis=0)

    labels_1 = l_p[indicies_0]
    labels_1= np.concatenate((labels_1, l_p[indicies_1]), axis=0)
    for i in range(2, 4):
        indicies = np.where(np.argmax(l_p,axis=1) == i)[0]
        print indicies.shape
        print "building"
        gmm = GaussianMixture(n_components=1).fit(d_p[indicies])
        '''

        num = 0
        d_fin = []
        l_fin = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x,y in zip(data_3[indicies],labels_3[indicies]):
                o = gmm.predict(x.reshape(1,-1))
                if o[0].item() is  0:
                    d_fin.append(x)
                    l_fin.append(y)
                    num = num +1
        print num
        gmm = GaussianMixture(n_components=1).fit(d_fin)
        d2, _ = gmm.sample(len(d_fin))
        l2 = l_fin
        '''
        print "sampling"
        d2,_ = gmm.sample(indicies.shape[0])
        l2 = l_p[indicies]
        print d2.shape
        print data_1.shape
        data_1 = np.concatenate((data_1, d2), axis=0)
        labels_1 =np.concatenate((labels_1, l2), axis=0)
    print "last data set"
    '''
    for i in range(0,1):
        indicies = np.where(np.argmax(labels_2,axis=1) == i)[0]

        gmm = GaussianMixture(n_components=1).fit(data_2[indicies])


        d2,_ = gmm.sample(indicies.shape[0])
        l2 = labels_2[indicies]
        data_1 = np.concatenate((data_1, d2), axis=0)
        labels_1 =np.concatenate((labels_1, l2), axis=0)
    '''
    sess = tf.Session()



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
    print data_1.shape
    print labels_1.shape
    print min(np.argmax(labels_1,axis=1))
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
                r = sess.run([acc], feed_dict={X:DATA[150:],Y:LABELS[150:]})
                output.append(r[0])
                print r

    OUTPUT.append(output)
    np.savetxt(filename, OUTPUT, delimiter=',')

    #11 - (48%)