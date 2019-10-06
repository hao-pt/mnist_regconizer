import numpy as np

from testCases_opti import *

def random_minibatches(X, Y, mini_batch_size = 64, seed = 0):
    '''
    Purpose: Shuffle dataset to generate mini-batches used for each epoches
    Input:
        X: input layer with shape [nx, m]
        Y: label layer with shape [#classes, m]
        mini_batch_size: the size of each mini_batche. Default: 64.
        seed: Set seed to get different permutation for each epoch
    Output:
        minibatches: list contains all shuffled mini batches.
    '''
    # Get number of examples
    m = X.shape[1]
    minibatches = []
    
    # Set seed
    np.random.seed(seed)
    
    # Permute index
    rand_index = list(np.random.permutation(m))
    
    # Shuffle dataset
    shuffle_X, shuffle_Y = X[:, rand_index], Y[:, rand_index]
    
    # Compute number of minibatch
    no_minibatches = int(m/mini_batch_size)

    # Loop over no_minibatch and append each minibatch to list
    for i in range(no_minibatches):
        # Slice each minibatch
        minibatch_X = shuffle_X[:, i*mini_batch_size : i*mini_batch_size + mini_batch_size]
        minibatch_Y = shuffle_Y[:, i*mini_batch_size : i*mini_batch_size + mini_batch_size]
        # Packing
        minibatch = (minibatch_X, minibatch_Y)
        
        # Append to list
        minibatches.append(minibatch)
        
    # Handle a remaining minibatch if the last minibatch is less than minibatch_size
    if m % mini_batch_size != 0:
        # Slice each minibatch
        minibatch_X = shuffle_X[:, no_minibatches*mini_batch_size:m]
        minibatch_Y = shuffle_Y[:, no_minibatches*mini_batch_size:m]
        # Packing
        minibatch = (minibatch_X, minibatch_Y)
        
        # Append to list
        minibatches.append(minibatch)
        
    return minibatches

def softmax(Z):
    """
    Perform Softmax activation function (stable version). We use stable version of softmax
    to avoid overflow in case of large exponents.
    Refer: https://machinelearningcoban.com/2017/02/17/softmax/#-softmax-regression-cho-mnist
    
    Args:
        Z: Output of linear function. (#numclass, Z)

    Returns:
        A: Output of softmax function. Same shape as Z
        cache: it is Z which is useful for backprop
    """
    cache = Z
    t = np.exp(Z - np.amax(Z, axis=0, keepdims=True))
    A = t / np.sum(t, axis=0)

    assert(A.shape==Z.shape)

    return A, cache

def softmax_backward(Y, activation_cache):
    """
    Perform backprop for single softmax activation unit [Linear <- Activation] for last layer
    Ref: https://deepnotes.io/softmax-crossentropy#cross-entropy-loss

    Args:
        Y: the groundtruth categorial labels of shape (#classes, m).
        activation_cache: is an output of linear function Z which we stored in forward prop

    Returns:
        dZ: gradient of cost function with respect to Z. Shape shape as Y
    """
    m = Y.shape[1]
    Z = activation_cache    # (#class, m)
    A, _ = softmax(Z)     # compute softmax function
    # g_prime = np.diagflat(A) - np.dot(A, A.T)   # compute derivative of softmax function
    dZ = (A - Y)/m # compute dZ = dA * g_prime

    assert(Y.shape == dZ.shape)

    return dZ

def cross_entropy_cost(A, Y):
    """
    Compute the cross-entropy cost for multi-class problem
    Note: m denotes the number of examples
    
    Args:
        AL: the output of softmax function in the L-th layer. Shape (#classes, m)
        Y: the groundtruth categorial labels of shape (#classes, m).
    
    Returns:
        cost: cost function (singular value)
    """
    
    m = Y.shape[1]  # Number of examples
    labels = np.argmax(Y, axis=0).squeeze()    # positions of maximum value corresponding to the actual class for each example
    loss = -np.log(A[labels, range(m)])    # Compute loss using multi-dimension array indexing to get the actual probability for each example
    cost = np.sum(loss) / m     # Compute the overall cost
    return cost


def sigmoid(Z):
    """
    Perform Sigmoid activation function
    
    Args:
        Z: Output of linear function. 
    Returns:
        A: Output of sigmoid function. Same shape as Z
        cache: it is Z which is useful for backprop
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    assert(A.shape==Z.shape)

    return A, cache

def relu(Z):
    """
    Perform Relu activation function
    
    Args:
        Z: Output of linear function. 
    Returns:
        A: Output of relu function. Same shape as Z
        cache: it is Z which is useful for backprop
    """
    A = np.maximum(0, Z)
    cache = Z
    
    assert(A.shape==Z.shape)

    return A, cache

def relu_backward(dA, activation_cache):
    """
    Perform backprop for single Relu activation unit [Linear <- Activation]
    
    Args:
        dA: gradient of cost function with respect to A
        activation_cache: is an output of linear function Z which we stored in forward prop
    Returns:
        dZ: gradient of cost function with respect to Z. Shape shape as dA
    """
    Z = activation_cache 
    dZ = dA.copy() # clone
    dZ[Z <= 0] = 0 # apply mask mask of derivative of relu function into dA

    assert(dA.shape == dZ.shape)

    return dZ

def sigmoid_backward_last_layer(Y, activation_cache):
    """
    Perform backprop for single sigmoid activation unit [Linear <- Activation] for last layer L
    
    Args:
        Y: the groundtruth categorial labels of shape (#classes, m).
        activation_cache: is an output of linear function Z which we stored in forward prop
    Returns:
        dZ: gradient of cost function with respect to Z. Shape shape as Y
    """
    m = Y.shape[1]
    Z = activation_cache
    AL, _ = sigmoid(Z) # compute sigmoid funtion for last layer
    dZ = (AL - Y)/m     # compute dZ = dA * g'(Z)

    assert(Y.shape == dZ.shape)

    return dZ

def sigmoid_backward(dA, activation_cache):
    """
    Perform backprop for single sigmoid activation unit [Linear <- Activation]
    
    Args:
        dA: gradient of cost function with respect to A
        activation_cache: is an output of linear function Z which we stored in forward prop
    Returns:
        dZ: gradient of cost function with respect to Z. Shape shape as dA
    """
    Z = activation_cache
    A, _ = sigmoid(Z)
    g_prime = A * (1 - A) # compute derivative of sigmoid function
    dZ = (dA * g_prime) # compute dZ

    assert(dA.shape == dZ.shape)

    return dZ

def single_layer_forward(A_prev, W, b, activation_name):
    """
    Perform forward for single layer
    Note: m denotes the number of examples
    
    Args:
        A_prev (#units of previous layer, m): activation of the previous layer
        W (#units of current layer, #units of previous layer): weight matrix of current layer
        b (#units of current layer, 1): bias vector of current layer
        activation_name: the name of activation function (e.g. sigmoid or relu)
    Returns:
        A (#units of current layer, m): output activation of the current layer
        cache: Stores "linear_cache" and "activation_cache" that are useful for computing backprop
    """
    # Linear
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)

    # Activation
    if activation_name == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation_name == 'relu':
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache

def linear_backward(dZ, linear_cache, weight_decay=0):
    """
    Perform the linear part of backward propagation for a single layer (layer l) [Linear <- Activation]
    Note: m denotes number of examples
    
    Args:
        dZ (#units in layer l, m): gradient of cost function with respect to Z at layer l.
        linear_cache: store the tuple of 3 values (A_prev, W, b) computed through forward propagation for this layer l
        weight_decay: the regularizer hyperparameter (default: 0)

    Returns:
        dA_prev (#units in layer l-1, m): gradient of cost function with respect to A at layer l - 1. Same shape as A_prev.
        dW (#units in layer l, #units in layer l-1): gradient of cost function with respect to W at layer l. Same shape as  W.
        db (#units in layer l, 1): gradient of cost function with respect to b at layer l. Same shape as b.
    """

    # Unpack linear_cache
    A_prev, W, b = linear_cache

    m = A_prev.shape[1] # no. of examples

    dW = np.dot(dZ, A_prev.T) # (#units in layer l, m) .* (m, #units in layer l-1) -> (#units in layer l, #units in layer l-1)
    db = np.sum(dZ, axis=1, keepdims=True) # (#units in layer l, 1), keepdims remain the shape of (l,) as (l,1)
    if weight_decay != 0:
        dW = dW + (weight_decay/m)*W
    
    dA_prev = np.dot(W.T, dZ) # (#units in layer l-1, #units in layer l) .* (#units in layer l, m) -> (#units in layer l-1, m)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

def single_layer_backward(dA, single_layer_cache, activation_name, weight_decay=0):
    """
    Perform backward propagation for a single layer (layer l) [Activation <- Linear <- Activation]
    Note: m denotes number of examples
    
    Args:
        dA (#units in layer l, m): gradient of cost function with respect to A at layer l
        single_layer_cache: Stores "linear_cache" and "activation_cache" that are useful for computing backprop
        activation_name: the name of activation function (e.g. sigmoid or relu)
        weight_decay: the regularizer (default: 0)

    Returns:
        dA_prev (#units in layer l-1, m): gradient of cost function with respect to A at layer l - 1. Same shape as A_prev.
        dW (#units in layer l, #units in layer l-1): gradient of cost function with respect to W at layer l. Same shape as  W.
        db (#units in layer l, 1): gradient of cost function with respect to b at layer l. Same shape as b.
        dZ (#units in layer l, m): gradient of cost function with respect to Z at layer l. Same shape as dA.
    """
    # Unpack single_layer_cache
    linear_cache, activation_cache = single_layer_cache
    
    # [Activation <- dW, db <- Linear <- Activation]
    if activation_name == "relu":
        dZ = relu_backward(dA, activation_cache) # (#units in layer l, m)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, weight_decay)
    elif activation_name == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache) # (#units in layer l, m)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, weight_decay)

    assert(dZ.shape == dA.shape)

    return dA_prev, dW, db, dZ

def single_layer_backward_last_layer(Y, single_layer_cache, activation_name, weight_decay=0):
    """
    Perform backward propagation for a last layer (layer L) [Activation <- Linear <- Activation]
    Note: m denotes number of examples
    
    Args:
        Y: the groundtruth categorial labels of shape (#classes, m).
        single_layer_cache: Stores "linear_cache" and "activation_cache" that are useful for computing backprop
        activation_name: the name of activation function (e.g. sigmoid or relu)
        weight_decay: the regularizer (default: 0)
    Returns:
        dA_prev (#units in layer L-1, m): gradient of cost function with respect to A at layer L - 1. Same shape as A_prev.
        dW (#units in layer L, #units in layer L-1): gradient of cost function with respect to W at layer L. Same shape as  W.
        db (#units in layer L, 1): gradient of cost function with respect to b at layer L. Same shape as b.
        dZ (#units in layer L, m): gradient of cost function with respect to Z at layer L. Same shape as Y.
    """
    # Unpack single_layer_cache
    linear_cache, activation_cache = single_layer_cache
    
    # [Activation <- dW, db <- Linear <- Activation]
    if activation_name == "sigmoid":
        dZ = sigmoid_backward_last_layer(Y, activation_cache) # (#units in layer L, m)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, weight_decay)
    elif activation_name == 'softmax':
        dZ = softmax_backward(Y, activation_cache) # (#units in layer L, m)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, weight_decay)

    assert(dZ.shape == Y.shape)

    return dA_prev, dW, db, dZ


# X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
# mini_batches = random_minibatches(X_assess, Y_assess, mini_batch_size)

# print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
# print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
# print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
# print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
# print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
# print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
# print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))