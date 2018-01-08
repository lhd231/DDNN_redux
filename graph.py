import matplotlib.pyplot as plt
import numpy as np

ddnn = np.mean(np.loadtxt("./output_ddnn.txt",delimiter=','),axis=0)
sing = np.mean(np.loadtxt("./output_sing.txt",delimiter=','),axis=0)
gmm = np.mean(np.loadtxt("./output_sing_gmm.txt",delimiter=','),axis=0)

plot_ddnn, = plt.plot(ddnn,label='ddnn')
plot_sing, = plt.plot(sing,label='sing')
plot_gmm, = plt.plot(gmm,label='gmm')

print gmm.shape
print ddnn.shape
print sing.shape

print sing[9]
print gmm[9]
plt.legend([plot_ddnn, plot_sing, plot_gmm])

plt.show()