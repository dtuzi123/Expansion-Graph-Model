from keras.datasets import mnist
import time
from utils import *
from scipy.misc import imsave as ims
from utils import *
import random as random
from glob import glob
import os, gzip
import keras as keras
from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")

import matplotlib.pyplot as plt

def ReadFiles(file):
	f1 = open(file)
	myData = []
	cNames = f1.readlines()
	for i in range(0, len(cNames)):
		myData.append(float(cNames[i]))
	f1.close()
	myData = np.array(myData)
	return myData

vaeLoss1Arr = []
vaeLoss2Arr = []
ELBO1Arr = []
ELBO2Arr = []

discrepancy = "E:/LifelongVAE_Theoreticalanalysis/results/CrossDomain_Discrepancy0.txt"
klDivergence = "E:/LifelongVAE_Theoreticalanalysis/results/CrossDomain_KLDivergence0.txt"
sourceRisk = "E:/LifelongVAE_Theoreticalanalysis/results/CrossDomain_SourceRisk0.txt"
targetRisk = "E:/LifelongVAE_Theoreticalanalysis/results/CrossDomain_TargetRisk0.txt"

discrepancy = "E:/LifelongVAE_Theoreticalanalysis/results/CrossDomain_Discrepancy_Inverse20.txt"
klDivergence = "E:/LifelongVAE_Theoreticalanalysis/results/CrossDomain_Discrepancy_Inverse20.txt"
sourceRisk = "E:/LifelongVAE_Theoreticalanalysis/results/CrossDomain_SourceRisk_Inverse20.txt"
targetRisk = "E:/LifelongVAE_Theoreticalanalysis/results/CrossDomain_TargetRisk_Inverse20.txt"

vae1 = "E:/LifelongVAE_Theoreticalanalysis/results/IELBO_TwoModels_VAE_20.txt"
vae2 = "E:/LifelongVAE_Theoreticalanalysis/results/IELBO_TwoModels_IWVAE1_20.txt"
vae3 = "E:/LifelongVAE_Theoreticalanalysis/results/IELBO_TwoModels_IWVAE2_20.txt"


f1 = open(vae1)
beginData = []
cNames = f1.readlines()
for i in range(0, len(cNames)):
	beginData.append(float(cNames[i]))
f1.close()
vae1 = np.array(beginData)

f1 = open(vae2)
beginData = []
cNames = f1.readlines()
for i in range(0, len(cNames)):
	beginData.append(float(cNames[i]))
f1.close()
vae2 = np.array(beginData)

f1 = open(vae3)
beginData = []
cNames = f1.readlines()
for i in range(0, len(cNames)):
	beginData.append(float(cNames[i]))
f1.close()
vae3 = np.array(beginData)


vae1 = vae1[50:100]
vae2 = vae2[50:100]
vae3 = vae3[50:100]

vae1 = np.abs(vae1)
vae2 = np.abs(vae2)
vae3 = np.abs(vae3)

xArray = np.zeros(np.shape(vae1)[0])
for i in range(np.shape(vae1)[0]):
    xArray[i] = i*10


x1=range(1,21)
x2=range(11,21)

x11 = range(0,10)

x1 = [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
x2 = [11,12,13,14,15,16,17,18,19,20]

y = [10,11,12,13]
x = [0,1,2,3]

data = y

values = []
values.append(vae1)
values.append(vae2)
values.append(vae3)
values = np.swapaxes(values,0,1)

#values = values*-1

data = pd.DataFrame(values, xArray, columns=["ELBO-GR","IWELBO-GR-5","IWELBO-GR-50"])

sns.lineplot(data=data, palette="tab10", linewidth=2.0)

#plt.xticks(rotation=15)
plt.grid(True)
plt.xlabel('Training steps')
plt.ylabel('Risk')
#plt.title('Square loss estimated during the training')
plt.show()
