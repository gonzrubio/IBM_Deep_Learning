"""RNNs for language modeling.

For this example, we will simply use a sample of clean, non-annotated words
(with the exception of one tag -- , which is used for rare words such as
 uncommon proper nouns) for our model. This means that we just want to predict
what the next words would be, not what they mean in context or their classes
on a given sentence.
"""

import numpy as np
import reader
import tensorflow as tf
import time


########################################################################
#                                                                      #
#                         Penn Treebank dataset                        #
#                                                                      #
########################################################################

# mkdir data
# wget -q -O data/ptb.zip https://ibm.box.com/\
#     shared/static/z2yvmhbskc45xd2a9a4kkn6hg4g4kj5r.zip
# unzip -o data/ptb.zip -d data
# cp data/ptb/reader.py .

# Download and extract the simple-examples dataset

# wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
# tar xzf simple-examples.tgz -C data/


data_dir = "data/simple-examples/data"     # Data directory for our dataset.


########################################################################
#                                                                      #
#                         Model hyperparameters                        #
#                                                                      #
########################################################################

init_scale = 0.1        # Initial weight scale
learning_rate = 1.0     # Initial learning rate
max_grad_norm = 5       # Gradient clipping
num_layers = 2          # The number of layers in our model
num_steps = 20          # The total number of recurrence steps (sentence len).
hidden_size_l1 = 256    # The number of processing units.
hidden_size_l2 = 128    # The number of processing units.
max_epoch = 15          # The total number of epochs in training.
max_epoch_decay_lr = 4  # Maximum num_epochs trained with initial learning rate
decay = 0.5             # The decay for the learning rate.
keep_prob = 1           # Probability of keeping data in the Dropout Layer.
batch_size = 30         # The size for each batch of data.
vocab_size = 10000      # The size of our vocabulary (unique words).
embeding_vec_size = 200  # size of n-dimensional representation of a word.
is_training = 1          # Training flag to separate training from testing.


########################################################################
#                                                                      #
#                          Network structure                           #
#                                                                      #
########################################################################

# The number of LSTM cells are 2. To give the model more expressive power,
# we can add multiple layers of LSTMs to process the data. The output of
# the first layer will become the input of the second and so on.
# The recurrence steps is 20, that is, when our RNN is "Unfolded", the
# recurrence step is 20.
# the structure is like:
# 200 input units -> [200x200] Weight -> 200 Hidden units (first layer) ->
# [200x200] Weight matrix -> 200 Hidden units (second layer) ->
# [200] weight Matrix -> 200 unit output
# The input shape is [batch_size, num_steps], that is [30x20].
# It will turn into [30x20x200] after embedding, and then 20x[30x200]
# Each LSTM has 200 hidden units which is equivalent to the dimensionality
# of the embedding words and output.


########################################################################
#                                                                      #
#                          Training data                               #
#                                                                      #
########################################################################

# Train data is a list of words, of size 929589, represented by numbers,
# e.g. [9971, 9972, 9974, 9975,...]
# We read data as mini-batches of size b=30. Assume the size of each
# sentence is 20 words (num_steps = 20). Then it will take 𝑓𝑙𝑜𝑜𝑟(𝑁/(𝑏×ℎ)+1=1548
# iterations for the learner to go through all sentences once. Where N is the
# size of the list of words, b is batch size, and h is size of each sentence.
# So, the number of iterators is 1548.
# Each batch data is read from train dataset of size 600, and shape of [30x20].

# Reads and separate data into training data, validation data and testing sets.
raw_data = reader.ptb_raw_data(data_dir)
train_data, valid_data, test_data, vocab, word_to_id = raw_data


def id_to_word(id_list):
    """Convert id to word."""
    line = []
    for w in id_list:
        for word, wid in word_to_id.items():
            if wid == w:
                line.append(word)
    return line


print(id_to_word(train_data[0:100]))


# Lets just read one mini-batch now and feed our network:
itera = reader.ptb_iterator(train_data, batch_size, num_steps)
first_touple = itera.__next__()
_input_data, _targets = first_touple
_input_data.shape
_targets.shape
_input_data[0:3]
print(id_to_word(_input_data[0, :]))   # Three sentences in the input data.
print(id_to_word(_targets[0, :]))      # Three sentences in the target data.


########################################################################
#                                                                      #
#                            Embeddings                                #
#                                                                      #
########################################################################

# We use word2vec approach. It is, in fact, a layer in our LSTM network,
# where the word IDs will be represented as a dense representation before
# feeding to the LSTM.

# The embedded vectors also get updated during the training process of the
# deep neural network. We create the embeddings for our input data.
# Embedding_vocab is matrix of [10000x200] for all 10000 unique words.

# Embedding_lookup() finds the embedded values for our batch of 30x20 words.
# It goes to each row of input_data, and for each word in the row/sentence,
# finds the correspond vector in embedding_dic.

# It creates a [30x20x200] tensor, so, the first element of inputs (the first
# sentence), is a matrix of 20x200, which each row of it, is vector
# representing a word in the sentence.

embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                            embeding_vec_size,
                                            batch_input_shape=(batch_size,
                                                               num_steps),
                                            trainable=True,
                                            name="embedding_vocab")
inputs = embedding_layer(_input_data)


########################################################################
#                                                                      #
#                                 RNN                                  #
#                                                                      #
########################################################################

# Create 2 layer stacked LSTM.
lstm_cell_l1 = tf.keras.layers.LSTMCell(hidden_size_l1)
lstm_cell_l2 = tf.keras.layers.LSTMCell(hidden_size_l2)
stacked_lstm = tf.keras.layers.StackedRNNCells([lstm_cell_l1, lstm_cell_l2])

# The input should be a Tensor of shape:
# [batch_size, max_time, embedding_vector_size], in our case
# it would be (30, 20, 200)

layer = tf.keras.layers.RNN(stacked_lstm,
                            [batch_size, num_steps],
                            return_state=False,
                            stateful=True,
                            trainable=True)

# Initialize the states of the nework:
# For each LSTM, there are 2 state matrices, c_state and m_state.
# c_state and m_state represent "Memory State" and "Cell State".
# Each hidden layer, has a vector of size 30, which keeps the states.
# So, for 200 hidden units in each LSTM, we have a matrix of size [30x200]

init_state = tf.Variable(tf.zeros([batch_size, embeding_vec_size]),
                         trainable=False)
layer.inital_state = init_state

# The output of the stackedLSTM comes from 128 hidden_layer, and in each time
# step(=20), one of them get activated. We use the linear activation to map
# the 128 hidden layer to a [30X20 matrix]

outputs = layer(inputs)

# We now create densely-connected neural network layer that would reshape
# the outputs tensor from [30 x 20 x 128] to [30 x 20 x 10000].
dense = tf.keras.layers.Dense(vocab_size)
logits_outputs = dense(outputs)

print("shape of the output from dense layer: ",
      logits_outputs.shape)  # (batch_size, sequence_length, vocab_size)

# A softmax activation layers is also then applied to derive the probability of
# the output being in any of the multiclass(10000 in this case) possibilities.

activation = tf.keras.layers.Activation('softmax')
output_words_prob = activation(logits_outputs)
print("shape of the output from the activation layer: ",
      output_words_prob.shape)  # (batch_size, sequence_length, vocab_size)

print("The probability of observing words in t=0 to t=20",
      output_words_prob[0, 0:num_steps])


########################################################################
#                                                                      #
#                          Objective function                          #
#                                                                      #
########################################################################

def crossentropy(y_true, y_pred):
    """Cross entropy."""
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


loss = crossentropy(_targets, output_words_prob)
cost = tf.reduce_sum(loss / batch_size)


########################################################################
#                                                                      #
#                              Training                                #
#                                                                      #
########################################################################

# lr = tf.Variable(0.0, trainable=False)
optimizer = tf.keras.optimizers.SGD(lr=0.0, clipnorm=max_grad_norm)
model = tf.keras.Sequential()
model.add(embedding_layer)
model.add(layer)
model.add(dense)
model.add(activation)
model.compile(loss=crossentropy, optimizer=optimizer)
model.summary()

# Get all TensorFlow variables marked as "trainable".
tvars = model.trainable_variables
[v.name for v in tvars]

# Gradient
x = tf.constant(1.0)
y = tf.constant(2.0)
with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    g.watch(y)
    func_test = 2 * x * x + 3 * x * y

var_grad = g.gradient(func_test, x)  # Will compute to 10.0
print(var_grad)
var_grad = g.gradient(func_test, y)  # Will compute to 3.0
print(var_grad)

with tf.GradientTape() as tape:
    # Forward pass.
    output_words_prob = model(_input_data)
    # Loss value for this batch.
    loss = crossentropy(_targets, output_words_prob)
    cost = tf.reduce_sum(loss, axis=0) / batch_size

# Get gradients of loss wrt the trainable variables.
grad_t_list = tape.gradient(cost, tvars)
print(grad_t_list)

# Define the gradient clipping threshold
grads, _ = tf.clip_by_global_norm(grad_t_list, max_grad_norm)

# Create the training TensorFlow Operation through our optimizer
train_op = optimizer.apply_gradients(zip(grads, tvars))


########################################################################
#                                                                      #
#                                LSTM                                  #
#                                                                      #
########################################################################

class PTBModel(object):
    """lSTM RNN."""

    def __init__(self):
        """Set paremeters for ease of use."""
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_size_l1 = hidden_size_l1
        self.hidden_size_l2 = hidden_size_l2
        self.vocab_size = vocab_size
        self.embeding_vector_size = embeding_vec_size
        self._lr = 1.0

        # Initializing the model using keras Sequential API
        self._model = tf.keras.models.Sequential()

        # Creating the word embeddings layer and adding it to the sequence
        with tf.device("/gpu:0"):
            # Create the embeddings for our input data. Size is hidden size.
            # [10000x200]
            self._embedding_layer = \
                tf.keras.layers.Embedding(self.vocab_size,
                                          self.embeding_vector_size,
                                          batch_input_shape=(self.batch_size,
                                                             self.num_steps),
                                          trainable=True,
                                          name="embedding_vocab")
            self._model.add(self._embedding_layer)
        # Creating the LSTM cell and connect it with the RNN structure
        # Create the LSTM Cells.
        # This creates only the structure for the LSTM and has to be associated
        # with a RNN unit still.
        # The argument  of LSTMCell is size of hidden layer, that is, the
        # number of hidden units of the LSTM (inside A).
        # LSTM cell processes one word at a time and computes probabilities of
        # the possible continuations of the sentence.
        lstm_cell_l1 = tf.keras.layers.LSTMCell(hidden_size_l1)
        lstm_cell_l2 = tf.keras.layers.LSTMCell(hidden_size_l2)

        # By taking in the LSTM cells as parameters, the StackedRNNCells
        # function junctions the LSTM units to the RNN units.
        # RNN cell composed sequentially of stacked simple cells.
        stacked_lstm = tf.keras.layers.StackedRNNCells([lstm_cell_l1,
                                                        lstm_cell_l2])

        # Creating the input structure for our RNN
        # Input structure is 20x[30x200]
        # Considering each word is represended by a 200 dimentional vector,
        # and we have 30 batchs, we create 30 word-vectors of size [30xx2000]
        # The input structure is fed from the embeddings, which are filled in
        # by the input data
        # Feeding a batch of b sentences to a RNN:
        # In step 1,  first word of each of the b sentences (in a batch) is
        # input in parallel.
        # In step 2,  second word of each of the b sentences is input
        # in parallel.
        # The parallelism is only for efficiency.
        # Each sentence in a batch is handled in parallel, but the network sees
        # one word of a sentence at a time and does the computations
        # accordingly. All the computations involving the words of all
        # sentences in a batch at a given time step are done in parallel.

        # Instantiating our RNN model and setting stateful to True to feed
        # forward the state to the next layer

        self._RNNlayer = tf.keras.layers.RNN(stacked_lstm,[batch_size, num_steps],return_state=False,stateful=True,trainable=True)

        # Define the initial state, i.e., the model state for the very first data point
        # It initialize the state of the LSTM memory. The memory state of the network is initialized with a vector of zeros and gets updated after reading each word.
        self._initial_state = tf.Variable(tf.zeros([batch_size,embeding_vec_size]),trainable=False)
        self._RNNlayer.inital_state = self._initial_state
    
        ############################################
        # Adding RNN layer to keras sequential API #
        ############################################        
        self._model.add(self._RNNlayer)
        
        #self._model.add(tf.keras.layers.LSTM(hidden_size_l1,return_sequences=True,stateful=True))
        #self._model.add(tf.keras.layers.LSTM(hidden_size_l2,return_sequences=True))
        
        
        ####################################################################################################
        # Instantiating a Dense layer that connects the output to the vocab_size  and adding layer to model#
        ####################################################################################################
        self._dense = tf.keras.layers.Dense(self.vocab_size)
        self._model.add(self._dense)
 
        
        ####################################################################################################
        # Adding softmax activation layer and deriving probability to each class and adding layer to model #
        ####################################################################################################
        self._activation = tf.keras.layers.Activation('softmax')
        self._model.add(self._activation)

        ##########################################################
        # Instantiating the stochastic gradient decent optimizer #
        ########################################################## 
        self._optimizer = tf.keras.optimizers.SGD(lr=self._lr, clipnorm=max_grad_norm)
        
        
        ##############################################################################
        # Compiling and summarizing the model stacked using the keras sequential API #
        ##############################################################################
        self._model.compile(loss=self.crossentropy, optimizer=self._optimizer)
        self._model.summary()


    def crossentropy(self,y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    def train_batch(self,_input_data,_targets):
        #################################################
        # Creating the Training Operation for our Model #
        #################################################
        # Create a variable for the learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        # Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
        tvars = self._model.trainable_variables
        # Define the gradient clipping threshold
        with tf.GradientTape() as tape:
            # Forward pass.
            output_words_prob = self._model(_input_data)
            # Loss value for this batch.
            loss  = self.crossentropy(_targets, output_words_prob)
            # average across batch and reduce sum
            cost = tf.reduce_sum(loss/ self.batch_size)
        # Get gradients of loss wrt the trainable variables.
        grad_t_list = tape.gradient(cost, tvars)
        # Define the gradient clipping threshold
        grads, _ = tf.clip_by_global_norm(grad_t_list, max_grad_norm)
        # Create the training TensorFlow Operation through our optimizer
        train_op = self._optimizer.apply_gradients(zip(grads, tvars))
        return cost
        
    def test_batch(self,_input_data,_targets):
        #################################################
        # Creating the Testing Operation for our Model #
        #################################################
        output_words_prob = self._model(_input_data)
        loss  = self.crossentropy(_targets, output_words_prob)
        # average across batch and reduce sum
        cost = tf.reduce_sum(loss/ self.batch_size)

        return cost
    @classmethod
    def instance(cls) : 
        return PTBModel()



########################################################################################################################
# run_one_epoch takes as parameters  the model instance, the data to be fed, training or testing mode and verbose info #
########################################################################################################################
def run_one_epoch(m, data,is_training=True,verbose=False):

    #Define the epoch size based on the length of the data, batch size and the number of steps
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.
    iters = 0
    
    m._model.reset_states()
    
    #For each step and data point
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size, m.num_steps)):
        
        #Evaluate and return cost, state by running cost, final_state and the function passed as parameter
        #y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
        if is_training : 
            loss=  m.train_batch(x, y)
        else :
            loss = m.test_batch(x, y)
                                   

        #Add returned cost to costs (which keeps track of the total costs for this epoch)
        costs += loss
        
        #Add number of steps to iteration counter
        iters += m.num_steps

        #if verbose and step % (epoch_size // 10) == 10:
            #print("Itr %d of %d, perplexity: %.3f speed: %.0f wps" % (step , epoch_size, np.exp(costs / iters), iters * m.batch_size / (time.time() - start_time)))

    # Returns the Perplexity rating for us to keep track of how the model is evolving
    return np.exp(costs / iters)


# Reads the data and separates it into training data, validation data and testing data
raw_data = reader.ptb_raw_data(data_dir)
train_data, valid_data, test_data, _, _ = raw_data


# Instantiates the PTBModel class
m=PTBModel.instance()   
K = tf.keras.backend 
for i in range(max_epoch):
    # Define the decay for this epoch
    lr_decay = decay ** max(i - max_epoch_decay_lr, 0.0)
    dcr = learning_rate * lr_decay
    m._lr = dcr
    K.set_value(m._model.optimizer.learning_rate,m._lr)
    #print("Epoch %d : Learning rate: %.3f" % (i + 1, m._model.optimizer.learning_rate))
    # Run the loop for this epoch in the training mode
    train_perplexity = run_one_epoch(m, train_data,is_training=True,verbose=True)
    #print("Epoch %d : Train Perplexity: %.3f" % (i + 1, train_perplexity))
        
    # Run the loop for this epoch in the validation mode
    valid_perplexity = run_one_epoch(m, valid_data,is_training=False,verbose=False)
    #print("Epoch %d : Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    
# Run the loop in the testing mode to see how effective was our training
test_perplexity = run_one_epoch(m, test_data,is_training=False,verbose=False)
print("Test Perplexity: %.3f" % test_perplexity)








