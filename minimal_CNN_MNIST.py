"""
Pure Python CNN Neural Network.

Based on Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition.
by Andrej Karpathy, Justin Johnson

Further developed by Pavel Krolevets @ Shanghai Jiao Tong University. e-mail: pavelkrolevets@sjtu.edu.cn

"""
# A bit of setup
import numpy as np
import matplotlib.pyplot as plt
import struct
import sys
#from Convolution_pure_python import CONV_forward, CONV_backward


# Function to read MNIST original idx files to nd.array
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

# prepering the MNIST data
# 1. train data
X_train = read_idx('./data/data_train_X')
X_train = (X_train - 255)/255
X_train = X_train[:5000,:,:]

Y_train = read_idx('./data/data_train_labels')
Y_train = Y_train[:5000]
# 2. test data
X_test = read_idx('./data/data_test_X')
X_test = (X_test - 255)/255
X_test = X_test[:50,:,:]

Y_test = read_idx('./data/data_test_Y')
Y_test = Y_test[:50]

def CONV_forward(input, W, b, stride):
    stepsH = ((input.shape[0] - W.shape[0]) / stride + 1)
    stepsW = ((input.shape[1] - W.shape[1]) / stride + 1)
    if stepsW.is_integer() == False:
        sys.exit("Spatial size wrong. Change stride or filter size!")
    V = np.zeros((int(stepsH), int(stepsW)))
    stepk = 0
    for k in range(V.shape[0]):
        stepm = 0
        for m in range(V.shape[1]):
            V[m, k] = np.sum(input[stepm: stepm+W.shape[0],
                                    stepk: stepk+W.shape[1]]*W) + b
            stepm += stride
        stepk += stride
    cache = (input, W, b)
    return V, cache

def CONV_backward(dV, cache, stride):
    (X, W, b) = cache

    X_h, X_w = X.shape
    W_h, W_w = W.shape

    dX = np.zeros(X.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    stepk = 0
    for k in range(dV.shape[0]):
        stepm = 0
        for m in range(dV.shape[1]):
            dW += X[stepm: stepm+W_h,
                        stepk: stepk+W_w]*dV[m,k]

            dX[stepm: stepm + W_h,
                    stepk: stepk + W_w] += W*dV[m,k]

            db += dV[m,k]

            stepm += stride
        stepk += stride

    return dX, dW


# Train a Linear Classifier

# initialize parameters randomly
W1 = 0.01 * np.random.randn(7, 7)
b1 = np.zeros((1, 1))

W2 = 0.01 * np.random.randn(22*22, 10)
b2 = np.zeros((1, 10))

# some hyperparameters
step_size = 0.01
reg = 1e-3  # regularization strength
show_loss = np.zeros(0)
# gradient descent loop

num_examples = X_train.shape[0]


for i in range(10000):
    # model

    Conv1, cache = CONV_forward(X_train[0,:,:], W1, b1, stride=1)
    Conv1 = np.maximum(0, np.reshape(Conv1, (1, 22*22)))

    output = np.dot(Conv1,W2)+b2

    # compute the class probabilities
    exp_scores = np.exp(output)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]
    #print(probs[:,:,labels])
    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[:, Y_train[0]])
    data_loss = np.sum(corect_logprobs)

    reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss

    if i % 10 == 0:
        print ("iteration %d: loss %f" % (i, loss))
        #step_size *= 0.95
        show_loss = np.append(show_loss, loss)

    # compute the gradient on scores
    dscores = probs
    dscores[:, Y_train[0]] -= 1
    #dscores /= num_examples

    # backpropate the gradient to the parameters (W,b)

    dW2 = np.dot(Conv1.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)

    dConv1 = np.dot(dscores, W2.T)
    dConv1 [dConv1 <=0] = 0
    dConv1 = dConv1.reshape((22,22))

    dX, dW1 = CONV_backward(dConv1, cache, 1)
    db1 = np.sum(dConv1)
    db1 = np.reshape(db1, (1,1))
    # add regularization gradient contribution
    dW2 += reg * W2
    dW1 += reg * W1

    # perform a parameter update

    W2 += -step_size * dW2
    b2 += -step_size * db2

    W1 += -step_size * dW1
    b1 += -step_size * db1

# # evaluate training set accuracy
# scores_layer1 = np.maximum(0, (np.dot(X_train, W1) + b1))
# scores_layer2 = np.maximum(0, (np.dot(scores_layer1, W2) + b2))
# scores = np.dot(scores_layer2, W3) + b3
# predicted_class = np.argmax(scores, axis=1)
# print('training accuracy: %.2f' % (np.mean(predicted_class == Y_train)))

# # evaluate test set accuracy
# scores_layer1 = np.maximum(0, (np.dot(X_test, W1) + b1))
# scores_layer2 = np.maximum(0, (np.dot(scores_layer1, W2) + b2))
# scores = np.dot(scores_layer2, W3) + b3
#
# predicted_class = np.argmax(scores, axis=1)
# print ('test accuracy: %.2f' % (np.mean(predicted_class == Y_test)))

plt.plot(show_loss) # plotting by columns
plt.show()

# # saving the model
# np.save('./model/w3.npy', W3)
# np.save('./model/w2.npy', W2)
# np.save('./model/w1.npy', W1)
# np.save('./model/b3.npy', b3)
# np.save('./model/b2.npy', b2)
# np.save('./model/b1.npy', b1)

