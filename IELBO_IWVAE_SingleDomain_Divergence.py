import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow import keras
import numpy as np
import os
import argparse
import datetime
import time
import sys
sys.path.insert(0, './src')
import utils
import iwae1
import iwae2
import TaskClassifier
from data_hand import *
from keras.utils import to_categorical

# TODO: control warm-up from commandline
parser = argparse.ArgumentParser()
parser.add_argument("--stochastic_layers", type=int, default=2, choices=[1, 2], help="number of stochastic layers in the model")
parser.add_argument("--n_samples", type=int, default=50, help="number of importance samples")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=-1,
                    help="numper of epochs, if set to -1 number of epochs "
                         "will be set based on the learning rate scheme from the paper")
parser.add_argument("--objective", type=str, default="iwae_elbo", choices=["vae_elbo", "iwae_elbo", "iwae_eq14", "vae_elbo_kl"])
parser.add_argument("--gpu", type=str, default='1', help="Choose GPU")
args = parser.parse_args()
print(args)

# ---- string describing the experiment, to use in tensorboard and plots
string = "main_{0}_{1}_{2}".format(args.objective, args.stochastic_layers, args.n_samples)

# ---- set the visible GPU devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ---- dynamic GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ---- set random seeds
np.random.seed(123)
tf.random.set_seed(123)

# ---- number of passes over the data, see bottom of page 6 in [1]
if args.epochs == -1:
    epochs = 0
    learning_rate_dict = {}

    for i in range(8):
        learning_rate = 0.001 * 10**(-i/7)
        learning_rate_dict[epochs] = learning_rate
        epochs += 3 ** i

else:
    epochs = args.epochs
    learning_rate_dict = {}
    learning_rate_dict[0] = 0.0001

# ---- load data
(Xtrain, ytrain), (Xtest, ytest) = keras.datasets.mnist.load_data()
Ntrain = Xtrain.shape[0]
Ntest = Xtest.shape[0]

# ---- reshape to vectors
Xtrain = Xtrain.reshape(Ntrain, -1) / 255
Xtest = Xtest.reshape(Ntest, -1) / 255

# ---- experiment settings
objective = args.objective
# n_latent = args.n_latent
# n_hidden = args.n_hidden
n_samples = args.n_samples
batch_size = args.batch_size
steps_pr_epoch = Ntrain // batch_size
total_steps = steps_pr_epoch * epochs

# ---- prepare tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = "/tmp/iwae/{0}/".format(string) + current_time + "/train"
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_log_dir = "/tmp/iwae/{0}/".format(string) + current_time + "/test"
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# ---- instantiate the model, optimizer and metrics
if args.stochastic_layers == 1:
    n_latent = [100]
    n_hidden = [200]
    model = iwae1.IWAE(n_hidden[0], n_latent[0])
    model2 = iwae1.IWAE(n_hidden[0], n_latent[0])
else:
    n_latent = [100, 50]
    n_hidden = [200, 100]
    model = iwae2.IWAE(n_hidden, n_latent)
    model2 = iwae2.IWAE(n_hidden, n_latent)

taskClassifier = TaskClassifier.TaskClassifier(n_hidden[0])

learning_rate_dict[0] = 0.0001
optimizer = keras.optimizers.Adam(learning_rate_dict[0], epsilon=1e-4)
optimizer2 = keras.optimizers.Adam(learning_rate_dict[0], epsilon=1e-4)
print("Initial learning rate: ", optimizer.learning_rate.numpy())

# ---- prepare plotting of samples during training
# use the same samples from the prior throughout training
pz = tfd.Normal(0, 1)
z = pz.sample([100, n_latent[-1]])

plt_epochs = list(2**np.arange(12))
plt_epochs.insert(0, 0)
plt_epochs.append(epochs-1)

# ---- binarize the test data
# we'll only do this once, while the training data is binarized at the
# start of each epoch
Xtest = utils.bernoullisample(Xtest)

# ---- do the training
start = time.time()
best = float(-np.inf)

#Split MNIST into Five tasks
mnistTrain, mnistTest, fashionTrain, fashionTest, imnistTrainX, imnistTestX, ifashionTrainX, ifashionTestX = GiveLifelongTasks_AcrossDomain()

mnistTest = utils.bernoullisample(mnistTest)
fashionTest = utils.bernoullisample(fashionTest)
imnistTestX = utils.bernoullisample(imnistTestX)
ifashionTestX = utils.bernoullisample(ifashionTestX)

RatedMnistX, _ = load_mnist("mnist")
RatedFashionX, _ = load_mnist("Fashion")
RatedMnistX = np.reshape(RatedMnistX, (-1, 28 * 28))
RatedFashionX = np.reshape(RatedFashionX, (-1, 28 * 28))

RatedMnistTrain = RatedMnistX[0:60000]
RatedMnistTest = RatedMnistX[60000:70000]

RatedFashionTrain = RatedFashionX[0:60000]
RatedFashionTest = RatedFashionX[60000:70000]

omnistTrainingSet,omnistTestingSet = Load_OMNIST(True)

RatedMnistTest = utils.bernoullisample(RatedMnistTest)
RatedFashionTest = utils.bernoullisample(RatedFashionTest)
omnistTestingSet = utils.bernoullisample(omnistTestingSet)

pz = tfd.Normal(0, 1)
step = 0
auxilaryX = 0
currentGenerated = 0

KLArr = []
discrepancyArr = []
targetArr = []
sourceArr = []

CaltechTraining,CaltechTesting = Load_Caltech101(True)

classData = np.concatenate((CaltechTraining,fashionTrain[0:int(np.shape(CaltechTraining)[0]/2)],mnistTrain[0:int(np.shape(CaltechTraining)[0]/2)]),axis=0)
classY = np.zeros((np.shape(classData)[0],2))
classY[0:np.shape(CaltechTraining)[0],0] = 1
classY[np.shape(CaltechTraining)[0]:np.shape(classData)[0],1] = 1
#classY[0:np.shape(CaltechTraining)[0],0] = 1
#classY[np.shape(CaltechTraining)[0]:np.shape(classData)[0],1] = 1


#testing

'''
n_examples3 = np.shape(classData)[0]
index3 = [i for i in range(n_examples3)]
np.random.shuffle(classData)
np.random.shuffle(classY)
classData = classData[index3]
classY = classY[index3]

for t in range(10):
    for i in range(1000):
        classImages = classData[i * batch_size:(i + 1) * batch_size]
        classLabel = classY[i * batch_size:(i + 1) * batch_size]
        res3 = taskClassifier.train_step(classImages, classLabel, 1, optimizer2, objective=objective)

batch1 = mnistTest[0:batch_size]
batch2 = CaltechTraining[0:batch_size]

p1,p1_ = taskClassifier(batch1)
print(p1_)
print("next")

p2,p2_ = taskClassifier(batch2)
print(p2_)
'''

taskCount = 3

def Select_Samples(currentX,classifier):
    mycount = np.shape(currentX)[0]
    mycount = int(mycount/batch_size)
    arr = []
    for i in range(mycount):
        batch = currentX[i*batch_size:(i+1)*batch_size]
        logits,pred = classifier(batch)
        predictions = tf.argmax(pred, 1, name="predictions")
        predictions = np.array(predictions)

        #print(pred)
        for j in range(batch_size):
            if predictions[j] == 0:
                arr.append(batch[j])

    arr = np.array(arr)
    return arr

for taskIndex in range(taskCount):
    if taskIndex == 0:
        currentX = CaltechTraining
        auxilaryX = CaltechTesting
        currentGenerated = CaltechTraining
    elif taskIndex == 1:
        currentX = omnistTrainingSet
        auxilaryX = np.concatenate((CaltechTesting,omnistTestingSet),axis=0)
    elif taskIndex == 2:
        currentX = fashionTrain
        auxilaryX = np.concatenate((CaltechTesting,omnistTestingSet),axis=0)
        auxilaryX = np.concatenate((auxilaryX,fashionTest),axis=0)
    elif taskIndex == 3:
        currentX = ifashionTrainX
    elif taskIndex == 5:
        currentX = RatedMnistTrain
    elif taskIndex == 6:
        currentX = RatedFashionTrain

    if taskIndex != 0:
        arr = []
        myCount = int(np.shape(currentX)[0]/batch_size)
        for i in range(myCount):
            z = pz.sample([batch_size, n_latent[-1]])
            generatedImages = model.generate_samples(z)
            for j in range(batch_size):
                arr.append(generatedImages[j])
        arr = np.array(arr)
        currentGenerated = arr
        currentX = np.concatenate((currentX,arr),axis=0)

    epochs = 500
    for epoch in range(epochs):

        # ---- binarize the training data at the start of each epoch
        Xtrain_binarized = utils.bernoullisample(currentX)
        auxilaryX_binarized = utils.bernoullisample(auxilaryX)

        n_examples = np.shape(Xtrain_binarized)[0]
        index = [i for i in range(n_examples)]
        np.random.shuffle(index)
        Xtrain_binarized = Xtrain_binarized[index]

        n_examples2 = np.shape(auxilaryX_binarized)[0]
        index2 = [i for i in range(n_examples2)]
        np.random.shuffle(index2)
        auxilaryX_binarized = auxilaryX_binarized[index2]

        n_examples3 = np.shape(classData)[0]
        index3 = [i for i in range(n_examples3)]
        np.random.shuffle(classData)
        np.random.shuffle(classY)
        classData = classData[index3]
        classY = classY[index3]

        counter = 0

        myCount = int(np.shape(Xtrain_binarized)[0]/batch_size)
        auxCount = int(n_examples2/batch_size)
        classCount = int(n_examples3/batch_size)

        for idx in range(myCount):
            step = step + 1
            step = step %100000

            batchImages = Xtrain_binarized[idx*batch_size:(idx+1)*batch_size]
            auxIdx = idx % auxCount
            auxilaryImages = auxilaryX_binarized[auxIdx*batch_size:(auxIdx+1)*batch_size]
            classIdx = idx % classCount
            classImages = classData[classIdx*batch_size:(classIdx+1)*batch_size]
            classLabel = classY[classIdx*batch_size:(classIdx+1)*batch_size]

            beta = 1.0
            res = model.train_step(batchImages, n_samples, beta, optimizer, objective=objective)
            res2 = model2.train_step(auxilaryImages, n_samples, beta, optimizer2, objective=objective)

            res3 = taskClassifier.train_step(classImages, classLabel, beta, optimizer2, objective=objective)

            if step % 200 == 0:
                # ---- monitor the test-set
                #test_res = model.val_step(Xtest, n_samples, beta)

                took = time.time() - start
                start = time.time()

                #print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                #      .format(epoch, epochs, 0, total_steps, res[objective].numpy(), test_res[objective], took))
                print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                      .format(epoch, epochs, 0, total_steps, res[objective].numpy(), 0, took))

        sampleCount = 5000
        #Calculate discrepancy
        if taskIndex == 0:
            selectedData = Xtrain_binarized
        else:
            selectedData = Select_Samples(Xtrain_binarized, taskClassifier)

        myTest1 = selectedData[0:sampleCount]
        myCount2 = int(np.shape(myTest1)[0]/batch_size)
        recoLoss1Sum = 0
        for kk in range(myCount2):
            batch = myTest1[kk*batch_size:(kk+1)*batch_size]
            x1 = model.GiveReconstruction(batch,1)
            x2 = model2.GiveReconstruction(batch,1)
            x1 = np.reshape(x1,(batch_size,-1))
            x2 = np.reshape(x2,(batch_size,-1))
            recoLoss1 = tf.keras.losses.binary_crossentropy(x1, x2+1e-07)
            recoLoss1 = tf.reduce_mean(recoLoss1)
            recoLoss1Sum = recoLoss1Sum+recoLoss1

        recoLoss1Sum = recoLoss1Sum / myCount2

        myTest2 = CaltechTesting[0:sampleCount]
        myCount2 = int(np.shape(myTest2)[0] / batch_size)
        recoLoss1Sum2 = 0
        for kk in range(myCount2):
            batch = myTest2[kk * batch_size:(kk + 1) * batch_size]
            x1 = model.GiveReconstruction(batch, 1)
            x2 = model2.GiveReconstruction(batch, 1)
            x1 = np.reshape(x1, (batch_size, -1))
            x2 = np.reshape(x2, (batch_size, -1))
            recoLoss2 = tf.keras.losses.binary_crossentropy(x1, x2 + 1e-07)
            recoLoss2 = tf.reduce_mean(recoLoss2)
            recoLoss1Sum2 = recoLoss1Sum2 + recoLoss2

        recoLoss1Sum2 = recoLoss1Sum2 / myCount2

        discrepancy = np.abs(recoLoss1Sum - recoLoss1Sum2)
        discrepancyArr.append(discrepancy)

        #Calculate KL divergence term
        myTest1 = selectedData[0:sampleCount]
        myTest2 = CaltechTesting[0:sampleCount]
        myCount2 = int(np.shape(myTest1)[0] / batch_size)
        recoLoss1Sum = 0
        klSum = 0
        for kk in range(myCount2):
            batch1 = myTest1[kk * batch_size:(kk + 1) * batch_size]
            batch2 = myTest2[kk * batch_size:(kk + 1) * batch_size]
            pz = tfd.Normal(0, 1)
            p1 = model.Give_Inference(batch1,n_samples)
            p2 = model.Give_Inference(batch2,n_samples)
            kl = tf.reduce_sum(tfd.kl_divergence(p1, pz), axis=-1)
            kl = tf.reduce_mean(kl)
            k2 = tf.reduce_sum(tfd.kl_divergence(p2, pz), axis=-1)
            k2 = tf.reduce_mean(k2)
            klterm = np.abs(kl-k2)
            klSum = klSum+klterm
        klSum = klSum / myCount2
        KLArr.append(klSum)

        #Calculate the source risk
        myTest1 = selectedData[0:sampleCount]
        myTest2 = CaltechTesting[0:sampleCount]
        myCount2 = int(np.shape(myTest1)[0] / batch_size)
        sourceRisk = 0
        targetRisk = 0
        sourceRisk = model.val_step(myTest1, n_samples,1)
        targetRisk = model.val_step(myTest2, n_samples,1)
        sourceRisk = sourceRisk[objective]
        targetRisk = targetRisk[objective]
        sourceRisk = float(sourceRisk)
        targetRisk = float(targetRisk)

        sourceArr.append(sourceRisk)
        targetArr.append(targetRisk)

lossArr1 = np.array(sourceArr).astype('str')
f = open("results/CrossDomain_SourceRisk_SingleDomain_Epoch" + str(0) + ".txt", "w", encoding="utf-8")
for i in range(np.shape(lossArr1)[0]):
    f.writelines(lossArr1[i])
    f.writelines('\n')
f.flush()
f.close()

lossArr1 = np.array(targetArr).astype('str')
f = open("results/CrossDomain_TargetRisk_SingleDomain_Epoch" + str(0) + ".txt", "w", encoding="utf-8")
for i in range(np.shape(lossArr1)[0]):
    f.writelines(lossArr1[i])
    f.writelines('\n')
f.flush()
f.close()

lossArr1 = np.array(discrepancyArr).astype('str')
f = open("results/CrossDomain_Discrepancy_SingleDomain_Epoch" + str(0) + ".txt", "w", encoding="utf-8")
for i in range(np.shape(lossArr1)[0]):
    f.writelines(lossArr1[i])
    f.writelines('\n')
f.flush()
f.close()

lossArr1 = np.array(KLArr).astype('str')
f = open("results/CrossDomain_KLDivergence_SingleDomain_Epoch" + str(0) + ".txt", "w", encoding="utf-8")
for i in range(np.shape(lossArr1)[0]):
    f.writelines(lossArr1[i])
    f.writelines('\n')
f.flush()
f.close()

# ---- save final weights
#model.save_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- load the final weights?
# model.load_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- test-set llh estimate using 5000 samples
