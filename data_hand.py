import numpy as np
import tensorflow as tf
import os
import gzip
import cv2
import keras as keras
import os
import scipy.io as scio
from Utils2 import *
#from scipy.misc import imsave as ims

def Give_InverseDataset(name):
    data_X, data_y = load_mnist(name)
    data_X = np.reshape(data_X, (-1, 28, 28))
    for i in range(np.shape(data_X)[0]):
        for k1 in range(28):
            for k2 in range(28):
                data_X[i, k1, k2] = 1.0 - data_X[i, k1, k2]

    data_X = np.reshape(data_X,(-1,28*28))
    return data_X

def GiveLifelongTasks_AcrossDomain():
    (train_images_nonbinary, y_train), (test_images_nonbinary, y_test) = tf.keras.datasets.mnist.load_data()

    train_images_nonbinary = train_images_nonbinary.reshape(train_images_nonbinary.shape[0], 28 * 28)
    test_images_nonbinary = test_images_nonbinary.reshape(test_images_nonbinary.shape[0], 28 * 28)

    '''
    y_train = tf.cast(y_train, tf.int64)
    y_test = tf.cast(y_test, tf.int64)
    '''

    train_images = train_images_nonbinary / 255.
    test_images = test_images_nonbinary / 255.

    '''
    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.
    '''

    mnistTrain = train_images
    mnistTest = test_images

    mnistName = "Fashion"
    data_X, data_y = load_mnist(mnistName)

    data_X = np.reshape(data_X,(-1,28*28))

    # data_X = np.expand_dims(data_X, axis=3)
    x_train = data_X[0:60000]
    x_test = data_X[60000:70000]

    '''
    x_train[x_train >= .5] = 1.
    x_train[x_train < .5] = 0.
    x_test[x_test >= .5] = 1.
    x_test[x_test < .5] = 0.
    '''
    fashionTrain = x_train
    fashionTest = x_test

    imnistX = Give_InverseDataset("mnist")
    ifashionX = Give_InverseDataset("Fashion")

    '''
    imnistX[imnistX >= .5] = 1.
    imnistX[imnistX < .5] = 0.
    ifashionX[ifashionX >= .5] = 1.
    ifashionX[ifashionX < .5] = 0.
    '''

    imnistTrainX = imnistX[0:60000]
    imnistTestX = imnistX[60000:70000]
    ifashionTrainX = ifashionX[0:60000]
    ifashionTestX = ifashionX[60000:70000]

    return mnistTrain,mnistTest,fashionTrain,fashionTest,imnistTrainX,imnistTestX,ifashionTrainX,ifashionTestX

def Load_Caltech101(isBinarized):
    dataFile = 'data/caltech101_silhouettes_28_split1.mat'
    data = scio.loadmat(dataFile)
    bc = 0

    trainingSet = data["train_data"]
    testingSet = data["test_data"]

    return trainingSet,testingSet

def Load_OMNIST(isBinarized):
    dataFile = 'data/omniglot.mat'
    dataFile = 'data/chardata.mat'
    data = scio.loadmat(dataFile)

    myData = data["data"]
    myData = myData.transpose(1, 0)

    if isBinarized == True:
        myData[myData >= .5] = 1.
        myData[myData < .5] = 0.

    trainingSet = myData
    testingSet = data["testdata"]
    testingSet = testingSet.transpose(1, 0)

    return trainingSet,testingSet

def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


def Split_dataset_by5(x,y):
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    labelArr1 = []
    labelArr2 = []
    labelArr3 = []
    labelArr4 = []
    labelArr5 = []

    n = np.shape(x)[0]
    for i in range(n):
        data1 = x[i]
        label1 = y[i]
        if label1[0] == 1 or label1[1] == 1:
            arr1.append(data1)
            labelArr1.append(label1)

        if label1[2] == 1 or label1[3] == 1:
            arr2.append(data1)
            labelArr2.append(label1)

        if label1[4] == 1 or label1[5] == 1:
            arr3.append(data1)
            labelArr3.append(label1)

        if label1[6] == 1 or label1[7] == 1:
            arr4.append(data1)
            labelArr4.append(label1)

        if label1[8] == 1 or label1[9] == 1:
            arr5.append(data1)
            labelArr5.append(label1)

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    arr3 = np.array(arr3)
    arr4 = np.array(arr4)
    arr5 = np.array(arr5)

    labelArr1 = np.array(labelArr1)
    labelArr2 = np.array(labelArr2)
    labelArr3 = np.array(labelArr3)
    labelArr4 = np.array(labelArr4)
    labelArr5 = np.array(labelArr5)
    return arr1,labelArr1,arr2,labelArr2,arr3,labelArr3,arr4,labelArr4,arr5,labelArr5
