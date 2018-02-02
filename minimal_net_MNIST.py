"""
Pure Python feed forward multi layer Neural Network.

Based on Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition.
by Andrej Karpathy, Justin Johnson

Further developed by Pavel Krolevets @ Shanghai Jiao Tong University. e-mail: pavelkrolevets@sjtu.edu.cn


"""
# A bit of setup
import numpy as np
import matplotlib.pyplot as plt
import struct

# Function to read MNIST original idx files to nd.array
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

# prepering the MNIST data
# 1. train data
X_train = read_idx('./data/data_train_X')
X_train = np.reshape(X_train, (60000, 28*28))
X_train = (X_train - 255)/255
X_train = X_train[:5000,:]

Y_train = read_idx('./data/data_train_labels')
Y_train = Y_train[:5000]
# 2. test data
X_test = read_idx('./data/data_test_X')
X_test = np.reshape(X_test, (10000, 28*28))
X_test = (X_test - 255)/255
X_test = X_test[:500,:]

Y_test = read_idx('./data/data_test_Y')
Y_test = Y_test[:500]


layer_size1 = 100
layer_size2 = 100
K = 10 # number of classes



# Train a Linear Classifier

# initialize parameters randomly
W1 = 0.01 * np.random.randn(784, layer_size1)
b1 = np.zeros((1, layer_size1))

W2 = 0.01 * np.random.randn(layer_size1, layer_size2)
b2 = np.zeros((1, layer_size2))

W3 = 0.01 * np.random.randn(layer_size2, K)
b3 = np.zeros((1, K))

# some hyperparameters
step_size = 0.1
reg = 1e-3  # regularization strength
show_loss = np.zeros(0)
# gradient descent loop
num_examples = X_train.shape[0]
for i in range(10000):

    # evaluate class scores, [N x K]
    hidden1 = np.maximum(0, (np.dot(X_train, W1) + b1))

    hidden2 = np.maximum(0, (np.dot(hidden1, W2) + b2))

    output = np.dot(hidden2, W3) + b3

    # compute the class probabilities
    exp_scores = np.exp(output)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]
    #print(probs[1,:])
    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(num_examples), Y_train])
    data_loss = np.sum(corect_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2) + 0.5 * reg * np.sum(W3 * W3)
    loss = data_loss + reg_loss

    if i % 500 == 0:
        print ("iteration %d: loss %f" % (i, loss))
        step_size *= 0.95
        show_loss = np.append(show_loss, loss)

    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples), Y_train] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters (W,b)
    dW3 = np.dot(hidden2.T, dscores)
    db3 = np.sum(dscores, axis=0, keepdims=True)

    dhidden2 = np.dot(dscores, W3.T)
    dhidden2 [hidden2 <=0] = 0

    dW2 = np.dot(hidden1.T, dhidden2)
    db2 = np.sum(dhidden2, axis=0, keepdims=True)

    dhidden1 = np.dot(dhidden2, W2.T)
    dhidden1[hidden1 <= 0] = 0

    dW1 = np.dot(X_train.T, dhidden1)
    db1 = np.sum(dhidden1, axis=0, keepdims=True)

    # add regularization gradient contribution
    dW3 += reg * W3
    dW2 += reg * W2
    dW1 += reg * W1

    # perform a parameter update


    W3 += -step_size * dW3
    b3 += -step_size * db3

    W2 += -step_size * dW2
    b2 += -step_size * db2

    W1 += -step_size * dW1
    b1 += -step_size * db1

# evaluate training set accuracy
scores_layer1 = np.maximum(0, (np.dot(X_train, W1) + b1))
scores_layer2 = np.maximum(0, (np.dot(scores_layer1, W2) + b2))
scores = np.dot(scores_layer2, W3) + b3
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == Y_train)))

# evaluate test set accuracy
scores_layer1 = np.maximum(0, (np.dot(X_test, W1) + b1))
scores_layer2 = np.maximum(0, (np.dot(scores_layer1, W2) + b2))
scores = np.dot(scores_layer2, W3) + b3

predicted_class = np.argmax(scores, axis=1)
print ('test accuracy: %.2f' % (np.mean(predicted_class == Y_test)))

plt.plot(show_loss) # plotting by columns
plt.show()

# saving the model
np.save('./model/w3.npy', W3)
np.save('./model/w2.npy', W2)
np.save('./model/w1.npy', W1)
np.save('./model/b3.npy', b3)
np.save('./model/b2.npy', b2)
np.save('./model/b1.npy', b1)

