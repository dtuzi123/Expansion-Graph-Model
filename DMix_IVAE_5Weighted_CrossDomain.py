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
from Utils2 import *
import iwae1
import iwae2
import DMix_Weighted
from data_hand import *
from scipy.misc import imsave as ims
from keras.utils import to_categorical

# TODO: control warm-up from commandline
parser = argparse.ArgumentParser()
parser.add_argument("--stochastic_layers", type=int, default=1, choices=[1, 2], help="number of stochastic layers in the model")
parser.add_argument("--n_samples", type=int, default=5, help="number of importance samples")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=-1,
                    help="numper of epochs, if set to -1 number of epochs "
                         "will be set based on the learning rate scheme from the paper")
parser.add_argument("--objective", type=str, default="iwae_elbo", choices=["vae_elbo", "iwae_elbo", "iwae_eq14", "vae_elbo_kl"])
parser.add_argument("--gpu", type=str, default='5', help="Choose GPU")
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


# ---- experiment settings
objective = args.objective
# n_latent = args.n_latent
# n_hidden = args.n_hidden
n_samples = args.n_samples
batch_size = args.batch_size
#steps_pr_epoch = Ntrain // batch_size
#total_steps = steps_pr_epoch * epochs
total_steps = 0

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
    #model = iwae1.IWAE(n_hidden[0], n_latent[0])
    model = DMix_Weighted.DMix_Weighted(n_hidden[0], n_latent[0])
    model.threshold = 100
    #model.Create_New_Component(myInput,0)
else:
    n_latent = [100, 50]
    n_hidden = [200, 100]
    model = iwae2.IWAE(n_hidden, n_latent)


myInput = tf.keras.layers.Input(shape=(28*28,))

learning_rate_dict[0] = 0.0001
optimizer = keras.optimizers.Adam(learning_rate_dict[0], epsilon=1e-4)
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
#Xtest = utils.bernoullisample(Xtest)

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
CaltechTraining,CaltechTesting = Load_Caltech101(True)

taskCount = 5

pz = tfd.Normal(0, 1)
step = 0
for taskIndex in range(taskCount):
    if taskIndex == 0:
        currentX = CaltechTraining
        currentNet = model.Create_New_Component(myInput, currentX[0:1000],True)
    elif taskIndex == 1:
        currentX = omnistTrainingSet
        model.SetTranable(False)
        currentNet = model.Create_New_Component(myInput, currentX[0:1000], False)
    elif taskIndex == 2:
        currentX = fashionTrain
        model.SetTranable(False)
        currentNet = model.Create_New_Component(myInput, currentX[0:1000], False)
    elif taskIndex == 3:
        currentX = mnistTrain
        model.SetTranable(False)
        currentNet = model.Create_New_Component(myInput, currentX[0:1000], False)
    elif taskIndex == 4:
        currentX = ifashionTrainX
        model.SetTranable(False)
        currentNet = model.Create_New_Component(myInput, currentX[0:1000], False)

    model.currentIndex = taskIndex
    epochs = 500
    for epoch in range(epochs):

        # ---- binarize the training data at the start of each epoch
        Xtrain_binarized = utils.bernoullisample(currentX)

        n_examples = np.shape(Xtrain_binarized)[0]
        index = [i for i in range(n_examples)]
        np.random.shuffle(index)
        Xtrain_binarized = Xtrain_binarized[index]
        counter = 0

        myCount = int(np.shape(Xtrain_binarized)[0]/batch_size)

        print(taskIndex)
        for idx in range(myCount):
            step = step + 1
            step = step %100000

            batchImages = Xtrain_binarized[idx*batch_size:(idx+1)*batch_size]
            beta = 1.0
            #res = model.train_step(batchImages, n_samples, beta, optimizer, objective=objective)
            with tf.GradientTape() as tape:
                if currentNet.IsBasic == True:
                    res = currentNet.Build_BasicNode(batchImages, n_samples)
                else:
                    b = 0
                    res = currentNet.Build_NormalNode(batchImages,model.BasicNodeArr,model.basic_number,model.Componentweights,model.n_samples)

                loss = -res

            trainable_weights = currentNet.trainable_weights
            grads = tape.gradient(loss, trainable_weights)
            optimizer.apply_gradients(zip(grads, trainable_weights))

            model.currentNode.logLikelihood = res

            if step % 200 == 0:
                # ---- monitor the test-set
                #test_res = model.val_step(Xtest, n_samples, beta)

                took = time.time() - start
                start = time.time()

                #print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                #      .format(epoch, epochs, 0, total_steps, res[objective].numpy(), test_res[objective], took))
                print("epoch {0}/{1}, step {2}/{3}, train ELBO: {4:.2f}, val ELBO: {5:.2f}, time: {6:.2f}"
                      .format(epoch, epochs, 0, total_steps, res, 0, took))

'''
model.save_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- load the final weights?
model.load_weights('/tmp/iwae/{0}/final_weights'.format(string))
'''

print("weight matrix")
print(model.ShownWeights)

L = 5000
index1,value1 = model.Evaluation(CaltechTesting,L)
index2,value2 = model.Evaluation(omnistTestingSet,L)
index3,value3 = model.Evaluation(fashionTest,L)
index4,value4 = model.Evaluation(mnistTest,L)
index5,value5 = model.Evaluation(ifashionTestX,L)

sum = value1+value2+value3+value4+value5
sum = sum/5.0
print(value1)
print(value2)
print(value3)
print(value4)
print(value5)
print(sum)

'''
# ---- save final weights
model.save_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- load the final weights?
# model.load_weights('/tmp/iwae/{0}/final_weights'.format(string))

# ---- test-set llh estimate using 5000 samples
test_elbo_metric = utils.MyMetric()
L = 5000

# ---- since we are using 5000 importance samples we have to loop over each element of the test-set


for i, x in enumerate(Xtest):
    res = model(x[None, :].astype(np.float32), L)
    test_elbo_metric.update_state(res['iwae_elbo'][None, None])
    if i % 200 == 0:
        print("{0}/{1}".format(i, Ntest))

test_set_llh = test_elbo_metric.result()
test_elbo_metric.reset_states()

print("Test-set {0} sample log likelihood estimate: {1:.4f}".format(L, test_set_llh))
'''