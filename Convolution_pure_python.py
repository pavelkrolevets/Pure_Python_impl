import struct
import numpy as np
import sys


X =  np.zeros((7,7))
X[:,:] = [[0,0,0,0,0,0,0],
            [0,1,2,2,1,0,0],
            [0,2,1,0,1,1,0],
            [0,2,0,2,2,1,0],
            [0,1,2,2,2,1,0],
            [0,1,1,0,1,2,0],
            [0,0,0,0,0,0,0]]

# X[:,:,1] = [[0,0,0,0,0,0,0],
#             [0,0,0,0,0,1,0],
#             [0,0,1,0,2,0,0],
#             [0,0,0,0,2,2,0],
#             [0,1,0,2,1,0,0],
#             [0,1,2,0,1,0,0],
#             [0,0,0,0,0,0,0]]
#
# X[:,:,2] = [[0,0,0,0,0,0,0],
#             [0,1,0,2,1,0,0],
#             [0,1,1,2,1,1,0],
#             [0,2,0,1,2,1,0],
#             [0,0,0,1,0,1,0],
#             [0,1,2,1,1,1,0],
#             [0,0,0,0,0,0,0]]

# initialize parameters randomly
W1 = np.zeros((3, 3, 2))
W1[:,:, 0] = [[ 0,-1, 0],
             [-1, 1, 1],
             [ 0, 1, 1]]

# W1[:,:,0] = [[-1,-1,-1],
#              [ 1, 0, 1],
#              [-1, 1, 0]]
#
# W1[:,:,0] = [[ 1,-1, 1],
#              [ 1, 0, 0],
#              [ 1,-1, 0]]

#W2 = np.zeros((3, 3, 3))
W1[:,:,1] = [[-1,-1, 1],
             [ 1, 0, 1],
             [ 0,-1, 1]]
# #
# W1[:,:,1] = [[ 0, 1, 1],
#              [-1, 0, 1],
#              [-1,-1, 0]]
#
# W1[:,:,1] = [[ 0,-1,-1],
#              [-1, 0, 0],
#              [ 0, 1,-1]]

b1 = np.zeros((1, 1, 2))
b1 += 1
print(b1)

print(W1.shape)


def CONV_forward(input, W, b, stride):
    stepsH = ((input.shape[0] - W.shape[0]) / stride + 1)
    stepsW = ((input.shape[1] - W.shape[1]) / stride + 1)

    if stepsW.is_integer() == False:
        sys.exit("Spatial size wrong. Change stride or filter size!")
    V = np.zeros((int(stepsH), int(stepsW), int(W.shape[2])))
    for layer in range(W.shape[2]):
        stepk = 0
        for k in range(V.shape[0]):
            stepm = 0
            for m in range(V.shape[1]):
                V[m, k, layer] = np.sum(input[stepm: stepm+W.shape[0],
                                        stepk: stepk+W.shape[1]]*W[:,:,layer]) + b[:,:, layer]
                stepm += stride
            stepk += stride
    cache = (input, W, b)
    return V, cache

V1, cache = CONV_forward(X, W1, b1, stride=1)
print('Volume', V1[:,:,1])

def CONV_backward(dV, cache, stride):
    (X, W, b) = cache

    X_h, X_w = X.shape
    W_h, W_w, W_layers = W.shape

    dX = np.zeros(X.shape)
    dW = np.zeros(W.shape)
    #db = np.zeros(b.shape)

    for layer in range(W_layers):
        stepk = 0
        for k in range(dV.shape[0]):
            stepm = 0
            for m in range(dV.shape[1]):
                dW[:,:,layer] += X[stepm: stepm+W_h,
                            stepk: stepk+W_w]*dV[m,k, layer]

                dX[stepm: stepm + W_h,
                        stepk: stepk + W_w] += W[:,:,layer]*dV[m,k,layer]

                #db += dV[m,k,layer]

                stepm += stride
            stepk += stride

    return dX, dW #, db

dX, dW = CONV_backward(V1, cache, 1)

print(dX, "\n", dW[:,:,1])


#
# """
# Convolution using tensor manipulations
# """
#
#
# def unfold(X, mode):
#     """This is a tool function to unfold a 3d tensor. Doesnt acept tensors of higher dimentions
#     mode = 0 - Mode of unfolding the tensor. Can have values 0,1,2 - as an input tensor has 3 dimensions.
#     This approach deffers from the matematical theory, where modes start from 1.
#     """
#     x, y, z = X.shape
#     #print('Shape of an input tensor: ', x, y, z)
#     if mode == 0:
#         G = np.zeros((x, y*z), dtype=float)
#         #print(G.shape)
#         k=0
#         for x_1 in range (z):
#             k=k+y
#             G[:,k-y:k] = X[:,:,x_1]
#     return G
#
# def conv_tensor_forward(input, W, b, stride, padding):
#     cache = W, b, stride, padding
#     h_filt, w_filt, d_filt, n_filt = W.shape
#     h_input, w_input, d_input = input.shape
#     h_out = (h_input - h_filt + 2 * padding) / stride + 1
#     w_out = (w_input - w_filt + 2 * padding) / stride + 1
#
#     if not h_out.is_integer() or not w_out.is_integer():
#         raise Exception('Invalid output dimension!')
#
#     out = np.zeros((int(h_out), int(w_out), int(n_filt)))
#
#     for layer in range(n_filt):
#         stepk = 0
#         for i in range(int(w_out)):
#             stepm = 0
#             for j in range(int(h_out)):
#                 X = input[stepm:stepm+h_filt, stepk:stepk+w_filt, :]
#                 X_unfold = unfold(X,mode=0)
#                 W1 = W[:,:,:,layer]
#                 W1_unfold = unfold(W1, mode=0)
#                 b1 = b[:,:,:,layer]
#
#                 out[j, i, layer] = np.sum(X_unfold*W1_unfold)+b1
#
#                 stepm += stride
#             stepk += stride
#
#     return out
#
# #X = X.reshape((-1,7,7,3))
#
# out = conv_tensor_forward(X, W1,b1, stride=1, padding=0)
# print(out[:,:,0])