import struct
import numpy as np
import sys


X =  np.zeros((7,7,3))
X[:,:,0] = [[0,0,0,0,0,0,0],
            [0,1,2,2,1,0,0],
            [0,2,1,0,1,1,0],
            [0,2,0,2,2,1,0],
            [0,1,2,2,2,1,0],
            [0,1,1,0,1,2,0],
            [0,0,0,0,0,0,0]]

X[:,:,1] = [[0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0],
            [0,0,1,0,2,0,0],
            [0,0,0,0,2,2,0],
            [0,1,0,2,1,0,0],
            [0,1,2,0,1,0,0],
            [0,0,0,0,0,0,0]]

X[:,:,2] = [[0,0,0,0,0,0,0],
            [0,1,0,2,1,0,0],
            [0,1,1,2,1,1,0],
            [0,2,0,1,2,1,0],
            [0,0,0,1,0,1,0],
            [0,1,2,1,1,1,0],
            [0,0,0,0,0,0,0]]

# initialize parameters randomly
W1 = np.zeros((3, 3, 3, 2))
W1[:,:,0, 0] = [[ 0,-1, 0],
             [-1, 1, 1],
             [ 0, 1, 1]]

W1[:,:,1,0] = [[-1,-1,-1],
             [ 1, 0, 1],
             [-1, 1, 0]]

W1[:,:,2,0] = [[ 1,-1, 1],
             [ 1, 0, 0],
             [ 1,-1, 0]]

#W2 = np.zeros((3, 3, 3))
W1[:,:,0,1] = [[-1,-1, 1],
             [ 1, 0, 1],
             [ 0,-1, 1]]

W1[:,:,1,1] = [[ 0, 1, 1],
             [-1, 0, 1],
             [-1,-1, 0]]

W1[:,:,2,1] = [[ 0,-1,-1],
             [-1, 0, 0],
             [ 0, 1,-1]]

b1 = np.zeros((1, 1, 1, 2))
b1 += 1
print(b1)

print(W1.shape)


def CONV(input, W, b, stride, layers):
    stepsH = ((input.shape[0] - W.shape[0]) / stride + 1)
    stepsW = ((input.shape[1] - W.shape[1]) / stride + 1)
    if stepsW.is_integer() == False:
        sys.exit("Spatial size wrong. Change stride or filter size!")

    V = np.zeros((int(stepsH), int(stepsW), layers))

    for layer in range(layers):
        stepk = 0
        for k in range(V.shape[0]):
            stepm = 0
            for m in range(V.shape[1]):
                V[m, k, layer] = np.sum(input[stepm: stepm+W.shape[0],
                                        stepk: stepk+W.shape[1],:]*W[:,:,:,layer]) + b[:,:,:,layer]
                stepm += stride
            stepk += stride
    return V

layers = 2


V1 = CONV(X, W1, b1, stride=1, layers=layers)

print('Volume', V1.shape)

#print('second layer', V2.shape)