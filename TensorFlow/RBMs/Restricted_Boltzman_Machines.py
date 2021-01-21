"""RBMs."""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import urllib.request
from IPython.display import Markdown, display
from PIL import Image
from tensorflow.keras.layers import Flatten
from utils import tile_raster_images


########################################################################
#                                                                      #
#                            Initialization                            #
#                                                                      #
########################################################################

# First, we have to load the utility file which contains different utility
# functions that are not connected in any way to the networks presented in the
# tutorials, but rather help in processing the outputs into a more
# understandable way.

with urllib.request.urlopen("https://cf-courses-data.s3.us.cloud-object-"
                            "storage.appdomain.cloud/"
                            "IBMDeveloperSkillsNetwork-DL0120EN-"
                            "SkillsNetwork/labs/Week4/"
                            "data/utils.py") as url:
    response = url.read()
target = open('utils.py', 'w')
target.write(response.decode('utf-8'))
target.close()


def printmd(string):
    """Display markdown."""
    display(Markdown('# <span style="color:red">'+string+'</span>'))


if not tf.__version__ == '2.2.0':
    printmd('<<<<<!!!!! ERROR !!!! please upgrade to TensorFlow 2.2.0,'
            'or restart your Kernel (Kernel->Restart & Clear Output)>>>>>')


########################################################################
#                                                                      #
#                             RBM Layrers                              #
#                                                                      #
########################################################################

# An RBM has two layers. The first layer of the RBM is called the visible (or
# input layer). Imagine that our toy example, has only vectors with 7 values,
# so the visible layer must have  𝑉=7  input nodes. The second layer is the
# hidden layer, which has  𝐻  neurons in our case. Each hidden node takes on
# values of either 0 or 1 (i.e.,  ℎ_𝑖=1 or  ℎ_𝑖=0), with a probability that is
# a logistic function of the inputs it receives from the other 𝑉 visible units,
# called for example,  𝑝(ℎ𝑖=1) . For our toy sample, we'll use 2 nodes in the
# hidden layer, so 𝐻=2 . Each node in the first layer also has a bias. We will
# denote the bias as  𝑣_𝑏𝑖𝑎𝑠 , and this single value is shared among the 𝑉
# visible units. The bias of the second is defined similarly as ℎ_𝑏𝑖𝑎𝑠, and
# this single value is shared among the  𝐻  hidden units.

num_nodes_visible = 7
num_nodes_hidden = 2
v_bias = tf.Variable(tf.zeros([num_nodes_visible], tf.float32))
h_bias = tf.Variable(tf.zeros([num_nodes_hidden], tf.float32))

# Weights among the input and hidden layer nodes (weight matrix).

W = tf.constant(np.random.normal(loc=0.0, scale=1.0,
                                 size=(num_nodes_visible,
                                       num_nodes_hidden)).astype(np.float32))


########################################################################
#                                                                      #
#                             Toy example                              #
#                                                                      #
########################################################################

# Assume that we have a trained RBM, and a input vector,
# such as [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]. Then the output of the
# forward pass would look like:

X = tf.constant([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], tf.float32)
v_state = X
h_bias = tf.constant([0.1, 0.1])
print("Input: ", v_state)
print("hb: ", h_bias)
print("W: ", W)

# Calculate the probabilities of turning the hidden units on:
h_prob = tf.nn.sigmoid(tf.matmul(v_state, W) + h_bias)
print("p(h|v): ", h_prob)

# Draw samples from the distribution:
h_state = tf.nn.relu(tf.sign(h_prob - tf.random.uniform(tf.shape(h_prob))))
print("h0 states:", h_state)

# Backward pass:
v_bias = tf.constant([0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1])
print("vb: ", h_bias)

# Calculate the probabilities of turning the visible units on:
v_prob = tf.nn.sigmoid(tf.matmul(h_state, tf.transpose(W)) + v_bias)
print("p(v|h): ", v_prob)

# Draw samples from the distribution:
v_state = tf.nn.relu(tf.sign(v_prob - tf.random.uniform(tf.shape(v_prob))))
print("v probability states: ", v_state)

# Given current state of hidden units and weights, what is the probability of
# generating [1. 0. 0. 1. 0. 0. 0.] in reconstruction phase, based on the above
# probability distribution function? Ithey will be bad since we havn't trained
# the network.

inp = X
v_probability = 1

for elm, p in zip(inp[0], v_prob[0]):
    if elm == 1:
        v_probability *= p
    else:
        v_probability *= (1-p)

# print("probability of generating X: ", v_probability.numpy())

########################################################################
#                                                                      #
#                                MNIST                                 #
#                                                                      #
########################################################################

# loading training and test data
mnist = tf.keras.datasets.mnist
(trX, trY), (teX, teY) = mnist.load_data()

# Flatten training data to match number of nodes in input layer 28*28
flatten = Flatten(dtype='float32')
trX = flatten(trX/255.0)
trY = flatten(trY/255.0)

# Visible and hidden layer bias weights. Arbitrarily chose 50 nodes in hidden.
num_visible_nodes = 28*28
num_hidden_nodes = 50
vb = tf.Variable(tf.zeros([num_visible_nodes]), tf.float32)
hb = tf.Variable(tf.zeros([num_hidden_nodes]), tf.float32)

# Weight tensor of size [num_visible_nodes, num_hidden_nodes]
W = tf.Variable(tf.zeros([num_visible_nodes, num_hidden_nodes]), tf.float32)

# Define visible layer
v0_state = tf.Variable(tf.zeros([784]), tf.float32)
print(tf.matmul([v0_state], W).shape)   # [1, num_hidden_nodes]

# Define hidden layer
h0_prob = tf.nn.sigmoid(tf.matmul([v0_state], W) + hb)


def hidden_layer(v0_state, W, hb):
    """Define a function to return only the generated hidden states."""
    # probabilities of the hidden units
    h0_prob = tf.nn.sigmoid(tf.matmul([v0_state], W) + hb)
    # sample_h_given_X
    h0_state = tf.nn.relu(tf.sign(h0_prob -
                                  tf.random.uniform(tf.shape(h0_prob))))
    return h0_state


h0_state = hidden_layer(v0_state, W, hb)
print("first 15 hidden states: ", h0_state[0][0:15])


def reconstructed_output(h0_state, W, vb):
    """Define a function to return only the generated visible states."""
    v1_prob = tf.nn.sigmoid(tf.matmul(h0_state, tf.transpose(W)) + vb)
    # sample_v_given_h
    v1_state = tf.nn.relu(tf.sign(v1_prob -
                                  tf.random.uniform(tf.shape(v1_prob))))
    return v1_state[0]


v1_state = reconstructed_output(h0_state, W, vb)
print("hidden state shape: ", h0_state.shape)
print("v0 state shape:  ", v0_state.shape)
print("v1 state shape:  ", v1_state.shape)


########################################################################
#                                                                      #
#                                 Training                             #
#                                                                      #
########################################################################

# The goal is to maximize the likelihood of our data being drawn from that
# distribution. In each epoch we calculate the sum of the squared differences
# between step 1 and step n, ie. the error shows the difference between the
# data and the reconstruction.

def error(v0_state, v1_state):
    """Compute the mean of elements across dimensions of a tensor."""
    return tf.reduce_mean(tf.square(v0_state - v1_state))


err = tf.reduce_mean(tf.square(v0_state - v1_state))
print("error", err)
print("error: ", error(v0_state, v1_state))
h1_state = hidden_layer(v1_state, W, hb)

# We use gibs sampling and contrastive divergence to update the weight matrix.
# W_k = W_(k-1) + alpha*deltaW_k. K=1 gives good results so we use that.

# Parameters
alpha = 0.01
epochs = 1
batchsize = 200
weights = []
errors = []
K = 1

# Create datasets
train_ds = tf.data.Dataset.from_tensor_slices((trX, trY)).batch(batchsize)

for epoch in range(epochs):
    for batch_number, dataxy in enumerate(train_ds):
        batch_x = dataxy[0]
        batch_y = dataxy[1]
        for i_sample in range(batchsize):
            for k in range(K):
                v0_state = batch_x[i_sample]
                h0_state = hidden_layer(v0_state, W, hb)
                v1_state = reconstructed_output(h0_state, W, vb)
                h1_state = hidden_layer(v1_state, W, hb)

                delta_W = tf.matmul(
                    tf.transpose([v0_state]), h0_state) - \
                    tf.matmul(tf.transpose([v1_state]), h1_state)
                W = W + alpha * delta_W

                vb = vb + alpha * tf.reduce_mean(v0_state - v1_state, 0)
                hb = hb + alpha * tf.reduce_mean(h0_state - h1_state, 0)

                v0_state = v1_state

            if i_sample == batchsize-1:
                err = error(batch_x[i_sample], v1_state)
                errors.append(err)
                weights.append(W)
                print('Epoch: %d' % epoch, "batch #: %i " % batch_number,
                      "of %i" % int(60e3/batchsize), "sample #: %i" % i_sample,
                      'reconstruction error: %f' % err)

plt.plot(errors)
plt.xlabel("Batch Number")
plt.ylabel("Error")
plt.show()

print(W.numpy())  # a weight matrix of shape (50,784)


########################################################################
#                                                                      #
#                               Learned features                       #
#                                                                      #
########################################################################

tile_raster_images(X=W.numpy().T, img_shape=(28, 28), tile_shape=(5, 10), tile_spacing=(1, 1))
import matplotlib.pyplot as plt
from PIL import Image
%matplotlib inline
image = Image.fromarray(tile_raster_images(X=W.numpy().T, img_shape=(28, 28) ,tile_shape=(5, 10), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')  

from PIL import Image
image = Image.fromarray(tile_raster_images(X =W.numpy().T[10:11], img_shape=(28, 28),tile_shape=(1, 1), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (4.0, 4.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')  

!wget -O destructed3.jpg  https://ibm.box.com/shared/static/vvm1b63uvuxq88vbw9znpwu5ol380mco.jpg
img = Image.open('destructed3.jpg')
img


# convert the image to a 1d numpy array
sample_case = np.array(img.convert('I').resize((28,28))).ravel().reshape((1, -1))/255.0

sample_case = tf.cast(sample_case, dtype=tf.float32)


hh0_p = tf.nn.sigmoid(tf.matmul(sample_case, W) + hb)
hh0_s = tf.round(hh0_p)

print("Probability nodes in hidden layer:" ,hh0_p)
print("activated nodes in hidden layer:" ,hh0_s)

# reconstruct
vv1_p = tf.nn.sigmoid(tf.matmul(hh0_s, tf.transpose(W)) + vb)

print(vv1_p)
#rec_prob = sess.run(vv1_p, feed_dict={ hh0_s: hh0_s_val, W: prv_w, vb: prv_vb})


img = Image.fromarray(tile_raster_images(X=vv1_p.numpy(), img_shape=(28, 28),tile_shape=(1, 1), tile_spacing=(1, 1)))
plt.rcParams['figure.figsize'] = (4.0, 4.0)
imgplot = plt.imshow(img)
imgplot.set_cmap('gray') 