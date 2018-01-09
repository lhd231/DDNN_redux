import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel, ttest_ind

ddnn_raw = np.loadtxt("./output_ddnn_old.txt",delimiter=',')[:,:10]
sing_raw = np.loadtxt("./output_sing_old.txt",delimiter=',')
gmm_raw = np.loadtxt("./output_sing_gmm.txt",delimiter=',')

ddnn = np.mean(ddnn_raw,axis=0)
sing = np.mean(sing_raw,axis=0)
gmm = np.mean(gmm_raw,axis=0)
ys = range(0,1000,100)
print ddnn.shape
print len(ys)
plot_ddnn, = plt.plot(ys,ddnn,label='Decentralized')
plot_sing, = plt.plot(ys,sing,label='Centralized')
plot_gmm, = plt.plot(ys,gmm,label='GMM')

last_DNN = ddnn_raw[:,9]
last_GMM = gmm_raw[:,9]
last_SING = sing_raw[:,9]
print gmm.shape
print ddnn.shape
print sing.shape
t,r = ttest_rel(last_DNN,last_GMM)
print sing[9]
print gmm[9]
plt.legend([plot_ddnn, plot_sing, plot_gmm], loc=4)
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy score")
print r

#plt.show()