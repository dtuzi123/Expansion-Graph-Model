from keras.datasets import mnist
import time
#from utils import *
from scipy.misc import imsave as ims
#from ops import *
#from utils import *
#from Utlis2 import *
import random as random
from glob import glob
import os, gzip
from glob import glob

'''
import keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
'''
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")

import matplotlib.pyplot as plt

x1 = [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
x2 = [11,12,13,14,15,16,17,18,19,20]

y = [10,11,12,13]
x = [0,1,2,3]

data = y
#values = values*-1

x = [289.32, 301.32,307.91,337.70]
x2 = [292.69, 296.41, 307.42,325.31]
y = [300.06, 299.22, 623.45,714.03]

fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

score = [1383, 474.31, 511.56, 568.54]
score = [436.23, 474.31, 534.56, 589.54]

Expert1 = [1,2,3,4,5,6] #50
Expert2 = [1,1,2,2,3,4]  #150
Expert3 = [1,1,1,1,2,2] #250
Expert4 = [1,1,1,1,1,1] #300

x = np.zeros(np.shape(Expert1)[0])
for i in range(np.shape(x)[0]):
    x[i] = i

x2 = np.zeros(np.shape(Expert3)[0])
for i in range(np.shape(x2)[0]):
    x2[i] = i


score = [86.96,93.51,97.24]
score2 = [5,4,2]
index = np.arange(len(score))
width = 0.8
ax2.bar(left=index, height=score, width=width, hatch='\\',color='#7B68EE')
ax.bar(left=index, height=score2, width=width, hatch='\\',color='#7B68EE')

ax.legend()
ax.set_xticklabels(['20', '40', '60'])
ax.set_xticks([0, 1, 2])
ax.set_ylabel('Number of basic nodes')

ax2.set_xticklabels(['20', '40', '60'])
ax2.set_xticks([0, 1, 2])
ax2.set_ylabel('Negative log-likelihood')
ax2.set_xlabel('Thresholds')

plt.show()
