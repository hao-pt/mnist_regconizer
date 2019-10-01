import numpy as np

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
    g_prime =  sigmoid(Z)[0] * (1 - sigmoid(Z)[0]) # compute derivative of sigmoid function
    dZ = dA *  g_prime # compute dZ

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

def single_layer_backward(dA, single_layer_cache, activation_name):
    """
    Perform backward propagation for a single layer (layer l) [Activation <- Linear <- Activation]
    Note: m denotes number of examples
    
    Args:
        dA (#units in layer l, m): gradient of cost function with respect to A at layer l
        single_layer_cache: Stores "linear_cache" and "activation_cache" that are useful for computing backprop
        activation_name: the name of activation function (e.g. sigmoid or relu)
    Returns:
        dA_prev (#units in layer l-1, m): gradient of cost function with respect to A at layer l - 1. Same shape as A_prev.
        dW (#units in layer l, #units in layer l-1): gradient of cost function with respect to W at layer l. Same shape as  W.
        db (#units in layer l, 1): gradient of cost function with respect to b at layer l. Same shape as b.
        dZ (#units in layer l, m): gradient of cost function with respect to Z at layer l. Same shape as dA.
    """
    # Unpack single_layer_cache
    linear_cache, activation_cache = single_layer_cache
    # Unpack linear_cache
    A_prev, W, b = linear_cache

    m = A_prev.shape[1] # no. of examples

    # [Activation <- dW, db <- Linear <- Activation]
    if activation_name == "relu":
        dZ = relu_backward(dA, activation_cache) # (#units in layer l, m)
    elif activation_name == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache) # (#units in layer l, m)

    dW = np.dot(dZ, A_prev.T)/m # (#units in layer l, m) .* (m, #units in layer l-1) -> (#units in layer l, #units in layer l-1)
    db = np.sum(dZ, axis=1, keepdims=True)/m # (#units in layer l, 1), keepdims remain the shape of (l,) as (l,1)
    dA_prev = np.dot(W.T, dZ) # (#units in layer l-1, #units in layer l) .* (#units in layer l, m) -> (#units in layer l-1, m)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dZ.shape == dA.shape)

    return dA_prev, dW, db, dZ