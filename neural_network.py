import numpy as np
import matplotlib.pyplot as plt
from nn_utils import *

from testCases_v4 import *
from dnn_app_utils_v3 import *
from testCases_reg import *
from opt_utils import *

class Neural_network():
    """Neural network class"""
    def __init__(self, layer_dims=[], L=0, params={}, cache_list=tuple(), grads={}, \
                 learning_rate=1e-4, no_epoches = 1000, X=[], Y=[], weight_decay=5e-4, \
                 loss_function_name='binary_cross_entropy', batch_size=64):
        """
        Initial attribute for class
        
        Args:
            layer_dims: list of number of units in each layer.
            L: number of layers (include input layer). (Defaults: 0)
            params: dictionary of parameters in network. For example: layer l has
                self.__params['W'+str(l)] = ...
                self.__params['b'+str(l)] = ...
                (Defaults: {})
            cache_list: stores all the single_layer_cache (linear_cache and activation_cache) of each single_layer_forward. 
                            (There are L - 1 single_layer_cache) (Defaults: ())
            grads: dictionay of gradients stored when backproping. For example: layer l has
                self.__grads['dA'+str(l)] = ...
                self.__grads['dW'+str(l)] = ...
                self.__grads['db'+str(l)] = ...
                self.__grads['dZ'+str(l)] = ...
            learning_rate: learning rate (Defaults: 1e-4)
            no_epoches: number of epoches (Defaults: 1000)
            X: input data of shape (input size, m) - numpy array.
            Y: the groundtruth categorial labels of shape (#classes, m).
            weight_decay: the regularizer hyperparameter (default: 5e-4)
            loss_function_name: loss function to use (e.g. binary_cross_entropy or softmax_cross_entropy)
            batch_size: batch size (Defaults: 64)

        Class Attributes:
            self.__layer_dims
            self.__L
            self.__params
            self.__cache_list
            self.__grads
            self.__cost: cost (singular value)
            self.__learning_rate
            self.__no_epoches
            self.__X
            self.__Y
            self.__weight_decay
            self.__loss_function_name
            self.batch_size
        """
        self.__layer_dims = layer_dims
        self.__L = max(len(layer_dims), L, (len(params)//2) + 1, len(cache_list)+1)
        self.__params = params
        self.__cache_list = cache_list
        self.__grads = grads
        self.__cost = 0.0
        self.__learning_rate = learning_rate
        self.__no_epoches = no_epoches
        self.__X = X
        self.__Y = Y
        self.__weight_decay = weight_decay # used in cost_function, single_layer_backward, linear_backward
        self.__loss_function_name = loss_function_name
        self.__batch_size = batch_size

    @property
    def layer_dims(self):
        """Getter function using property decorator"""
        return self.__layer_dims
    @layer_dims.setter
    def layer_dims(self, src):
        """Setter function using property decorator"""
        self.__layer_dims = src

    @property
    def L(self):
        """Getter function using property decorator"""
        return self.__L
    @L.setter
    def L(self, src):
        """Setter function using property decorator"""
        self.__L = src

    @property
    def params(self):
        """Getter function using property decorator"""
        return self.__params
    @params.setter
    def params(self, src):
        """Setter function using property decorator"""
        self.__params = src

    @property
    def cache_list(self):
        """Getter function using property decorator"""
        return self.__cache_list
    @cache_list.setter
    def cache_list(self, src):
        """Setter function using property decorator"""
        self.__cache_list = src
    
    @property
    def grads(self):
        """Getter function using property decorator"""
        return self.__grads
    @grads.setter
    def grads(self, src):
        """Setter function using property decorator"""
        self.__grads = src

    @property
    def learning_rate(self):
        """Getter function using property decorator"""
        return self.__learning_rate
    @learning_rate.setter
    def learning_rate(self, src):
        """Setter function using property decorator"""
        self.__learning_rate = src
        
    @property
    def no_epoches(self):
        """Getter function using property decorator"""
        return self.__no_epoches
    @no_epoches.setter
    def no_epoches(self, src):
        """Setter function using property decorator"""
        self.__no_epoches = src

    @property
    def X(self):
        """Getter function using property decorator"""
        return self.__X
    @X.setter
    def X(self, src):
        """Setter function using property decorator"""
        self.__X = src

    @property
    def Y(self):
        """Getter function using property decorator"""
        return self.__Y
    @Y.setter
    def Y(self, src):
        """Setter function using property decorator"""
        self.__Y = src

    @property
    def weight_decay(self):
        """Getter function using property decorator"""
        return self.__weight_decay
    @weight_decay.setter
    def weight_decay(self, src):
        """Setter function using property decorator"""
        self.__weight_decay = src
    
    @property
    def loss_function_name(self):
        """Getter function using property decorator"""
        return self.__loss_function_name
    @loss_function_name.setter
    def loss_function_name(self, src):
        """Setter function using property decorator"""
        self.__loss_function_name = src

    @property
    def batch_size(self):
        """Getter function using property decorator"""
        return self.__batch_size
    @batch_size.setter
    def batch_size(self, src):
        """Setter function using property decorator"""
        self.__batch_size = src

    def initialize_params(self):
        """
        Initialize parameters for neural network using He initalization method (He et al)
        
        Args:
            layer_dims: list of number of units in each layer (except input layer).
        Returns:
            self.__params: dictionary of initial parameters of the network
                W1 (layer_dims[1], layer_dims[0]): weight matrix of layer 1
                b1 (layer_dims[1], 1): bias vector of layer 1
                ...
                WL (layer_dims[L], layer_dims[L-1]): weight matrix of layer L
                bL (layer_dims[L], 1): bias vector of layer L
        """
        np.random.seed(3)
        for l in range(1, self.__L):
            self.__params['W'+str(l)] = np.random.randn(self.__layer_dims[l],self.__layer_dims[l-1]) * np.sqrt(2/self.__layer_dims[l - 1]) #/np.sqrt(self.__layer_dims[l-1]) 
            self.__params['b'+str(l)] = np.zeros((self.__layer_dims[l], 1))
            # Assert the shape
            assert(self.__params['W' + str(l)].shape == (self.__layer_dims[l], self.__layer_dims[l-1]))
            assert(self.__params['b' + str(l)].shape == (self.__layer_dims[l], 1))

        return self.__params
    
    def multi_layer_forward(self, X):
        """
        Perform multi-layer forward propagation: [Linear -> Relu] * (L-1) and [Linear -> Sigmoid]
        Note: m denotes the number of examples
        
        Args:
            self.__params: dictionary of initial parameters of the network
                W1 (layer_dims[1], layer_dims[0]): weight matrix of layer 1
                b1 (layer_dims[1], 1): bias vector of layer 1
                ...
                WL (layer_dims[L], layer_dims[L-1]): weight matrix of layer L
                bL (layer_dims[L], 1): bias vector of layer L
            X: data of shape (input size, m) - numpy array.
        Returns:
            AL: the output of activation function at the L-th layer. Shape (#classes, m)
            cache_list: stores all the single_layer_cache (linear_cache and activation_cache) of each single_layer_forward. 
                            (There are L - 1 single_layer_cache)
        """
        
        # Firstly, assign A = X
        A = X
        L = self.__L - 1 # exclude input layer
        
        cache_list = []

        # [Linear -> RELU]: From layer 1 to L - 1
        for l in range(1, L):
            A_prev = A
            A, single_layer_cache = single_layer_forward(A_prev, self.__params['W'+str(l)], self.__params['b'+str(l)], 'relu')
            cache_list.append(single_layer_cache)
        
        # [Linear -> Sigmoid]: at layer L
        AL, single_layer_cache = single_layer_forward(A, self.__params['W'+str(L)], self.__params['b'+str(L)], 'sigmoid')
        cache_list.append(single_layer_cache)

        return AL, cache_list

    def cost_function(self, AL, Y):
        """
        Compute the cross-entropy cost
        Note: m denotes the number of examples
        
        Args:
            AL: the output of activation function in the L-th layer. Shape (#classes, m)
            Y: the groundtruth categorial labels of shape (#classes, m).
        
        Returns:
            cost: cost function (singular value)
        """
        
        # Number of examples
        m = Y.shape[1]
        cost = -np.nansum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))/m

        # If Using regularization
        if self.__weight_decay != 0:
            L = self.L
            regularization = np.sum(np.sum(np.square(self.__params['W'+str(l)])) for l in range(1, L))
            cost += (self.__weight_decay / (2*m)) * regularization

        self.__cost = np.squeeze(cost) # make sure cost is singular value
        assert(self.__cost.shape == ())
        return self.__cost

    def multi_layer_backward(self, AL, Y):
        """
        Perform backward propagation for the whole network [Linear <- Relu] * (L-1) and [Linear <- Sigmoid]
        
        Args:
            AL: the output of activation function at the L-th layer. Shape (#classes, m)
            Y: the groundtruth categorial labels of shape (#classes, m).
            self.__cache_list: stores all the single_layer_cache (linear_cache and activation_cache) of each single_layer_forward. 
                            (There are L - 1 single_layer_cache)
                            For
        Returns:
            self.__grads: dictionay of gradients stored when backproping. For example: layer l has
                self.__grads['dA'+str(l)] = ...
                self.__grads['dW'+str(l)] = ...
                self.__grads['db'+str(l)] = ...
                self.__grads['dZ'+str(l)] = ...
        """
        
        m = Y.shape[1] # No.examples
        L = self.__L - 1 # No.layers exclude input layer

        # # Derivative of cost function respect to AL
        # dAL = (-Y/AL + (1-Y)/(1-AL))
        # self.__grads['dA'+str(L)] = dAL # Store in self.__grads

        # [Linear <- Sigmoid]: Backprop at last layer L
        #single_layer_backward(dAL, self.__cache_list[L-1], 'sigmoid') 
        self.__grads['dA'+str(L-1)], self.__grads['dW'+str(L)], self.__grads['db'+str(L)], self.__grads['dZ'+str(L)] = single_layer_backward_last_layer(Y, self.__cache_list[L-1], 'sigmoid', self.__weight_decay)
        
        # [Linear <- Relu] * (L-1): Backprop through L - 1 layer
        # Loop from L-1 to 1
        for l in reversed(range(L - 1)):
            single_layer_cache = self.__cache_list[l]
            self.__grads['dA'+str(l)], self.__grads['dW'+str(l+1)], self.__grads['db'+str(l+1)], self.__grads['dZ'+str(l+1)] = single_layer_backward(self.__grads['dA'+str(l+1)], single_layer_cache, 'relu', self.__weight_decay)
        
        return self.__grads

    def update_parameters(self):
        """
        Update parameters of the model using gradients computed by backprop
        
        Args:
            self.__params: dictionary of training parameters of the network
                W1 (layer_dims[1], layer_dims[0]): weight matrix of layer 1
                b1 (layer_dims[1], 1): bias vector of layer 1
                ...
                WL (layer_dims[L], layer_dims[L-1]): weight matrix of layer L
                bL (layer_dims[L], 1): bias vector of layer L
            self.__grads: dictionay of gradients stored when backproping. For example: layer l has
                self.__grads['dA'+str(l)] = ...
                self.__grads['dW'+str(l)] = ...
                self.__grads['db'+str(l)] = ...
                self.__grads['dZ'+str(l)] = ...
            self.__learning_rate: learning rate
        Returns:
            self.__params: dictionary of updated parameters of the network
        """
        L = self.__L - 1 #No.layers (exclude input layer)
        # Update W, b through each layer from 1 to L
        for l in range(L):
            self.__params['W'+str(l+1)] = self.__params['W'+str(l+1)] - self.__learning_rate * self.__grads['dW'+str(l+1)]
            self.__params['b'+str(l+1)] = self.__params['b'+str(l+1)] - self.__learning_rate * self.__grads['db'+str(l+1)]

        return self.__params

    def model(self, plot_learning_curve=False):
        """
        Train L layers neural network: [Linear -> Relu] * (L-1) and [Linear -> Sigmoid]
        
        1. Initialize parameters for training
        2. Iterate over no_epoches
            a. Forward propagation
            b. Compute cost function
            c. Backward propagation
            d. Update parameters (using parameter and gradients from backprop)
        3. Prediction

        Args:
            self.__X: data of shape (input size, m) - numpy array.
            self.__Y: the groundtruth categorial labels of shape (#classes, m).
            self.__no_epoches: number of epoches
            plot_learning_curve: Boolean value (Defaults: False). If True, this will plot learning curve of cost every 100 epoches
        Returns:
            params: dictionary of optimal parameters
            
        """
        
        X, Y = self.__X, self.__Y

        no_epoches = self.__no_epoches
        costs = [] # Store cost every 100 epoches for plotting

        # 1. Initialize parameters for training
        self.__params = self.initialize_params()
        
        # 2. Iterate over no_epoches
        for epoch in range(no_epoches):
            
            minibatches = random_minibatches(X, Y, self.__batch_size, seed=epoch)
            # Compute the number of minibatch
            no_minibatches = len(minibatches)
            cost = 0.0

            for minibatch in minibatches:
                # Unpack minibatch
                X_minibatch, Y_minibatch = minibatch
                # 2a. Forward propagation
                AL, self.__cache_list = self.multi_layer_forward(X_minibatch)
                # 2b. Compute cost function
                cost += self.cost_function(AL, Y_minibatch)/no_minibatches
                # 2c. Backward propagation
                self.__grads = self.multi_layer_backward(AL, Y_minibatch)
                # 2d. Update parameters (using parameter and gradients from backprop)
                self.__params = self.update_parameters()

            if epoch % 100 == 0:
                print("Epoch %d: cost = %f"%(epoch, cost))
                costs.append(cost)
        
        if plot_learning_curve:
            plt.plot(np.squeeze(costs))
            plt.xlabel('Iteration per hundred')
            plt.ylabel('cost')
            plt.title('Learning curve with learning rate = '+ str(self.__learning_rate))

        return self.__params
            
    def predict(self):
        """
        Predict results based on input data X

        Args:
            self.__X: data of shape (input size, m) - numpy array.
        
        Returns:
            Y_hat: the predicted results in form of one-hot vector. Shape (#classes, m)
        """
        AL, _ = self.multi_layer_forward(self.__X)
        
        Y_hat = np.zeros(AL.shape)

        for i in range(AL.shape[1]):
            if AL[0, i] > 0.5:
                Y_hat[0, i] = 1
            else:
                Y_hat[0, i] = 0
           
        return Y_hat

    def evaluate(self):
        """
        Evaluate predicted esults based on input data X and label Y

        Args:
            self.__X: data of shape (input size, m) - numpy array.
            self.__Y: the groundtruth categorial labels of shape (#classes, m).

        Returns:
            accuracy: the accuracy of model on X, Y dataset. (singular value)
        """
        Y = self.__Y
        Y_hat = self.predict()
        accuracy = np.mean(Y_hat == Y)
        return accuracy










            


        



# nn = Neural_network(layer_dims=[5, 4, 3])
# parameters = nn.initialize_params()
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# A_prev, W, b = linear_activation_forward_test_case()
# A, cache = single_layer_forward(A_prev, W, b, 'sigmoid')
# print(A)

# A, cache = single_layer_forward(A_prev, W, b, 'relu')
# print(A)


# X, parameters = L_model_forward_test_case_2hidden()
# nn = Neural_network()
# nn.L = len(parameters) // 2 + 1
# nn.params = parameters
# AL, caches = nn.multi_layer_forward(X)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))

# nn = Neural_network()
# Y, AL = compute_cost_test_case()
# print("cost = " + str(nn.cost_function(AL, Y)))

# dAL, linear_activation_cache = linear_activation_backward_test_case()

# dA_prev, dW, db, dZ = single_layer_backward(dAL, linear_activation_cache, activation_name = "sigmoid")
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")

# dA_prev, dW, db, dZ = single_layer_backward(dAL, linear_activation_cache, activation_name = "relu")
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

# nn = Neural_network()
# AL, Y_classes, caches = L_model_backward_test_case()
# nn.L = len(caches) + 1
# nn.cache_list = caches
# grads = nn.multi_layer_backward(AL, Y_classes)
# print('dW1 =', grads['dW1'])
# print('db1 =', grads['db1'])
# print('dA1 =', grads['dA1'])

# parameters, grads = update_parameters_test_case()
# nn = Neural_network()
# nn.params = parameters
# nn.grads = grads
# nn.L = len(parameters) // 2 + 1
# nn.learning_rate = 0.1

# parameters = nn.update_parameters()

# print ("W1 = "+ str(parameters["W1"]))
# print ("b1 = "+ str(parameters["b1"]))
# print ("W2 = "+ str(parameters["W2"]))
# print ("b2 = "+ str(parameters["b2"]))

"""Model test"""
# train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# # Explore your dataset 
# m_train = train_x_orig.shape[0]
# num_px = train_x_orig.shape[1]
# m_test = test_x_orig.shape[0]

# print ("Number of training examples: " + str(m_train))
# print ("Number of testing examples: " + str(m_test))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_x_orig shape: " + str(train_x_orig.shape))
# print ("train_y shape: " + str(train_y.shape))
# print ("test_x_orig shape: " + str(test_x_orig.shape))
# print ("test_y shape: " + str(test_y.shape))

# # Reshape the training and test examples 
# train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
# test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# # Standardize data to have feature values between 0 and 1.
# train_x = train_x_flatten/255.
# test_x = test_x_flatten/255.

# print ("train_x's shape: " + str(train_x.shape))
# print ("test_x's shape: " + str(test_x.shape))

# nn = Neural_network(layer_dims=[12288, 20, 7, 5, 1])
# nn.no_epoches = 1000
# nn.learning_rate = 0.0075
# nn.X = train_x
# nn.Y = train_y

# params = nn.model(plot_learning_curve=True)

# acc_train = nn.evaluate()

# nn.X = test_x
# nn.Y = test_y
# acc_test = nn.evaluate()
# print("Accuracy on training set:", acc_train)
# print("Accuracy on test set:", acc_test)

# plt.show()

"""Cost with Ref test"""
# A3, Y_assess, parameters = compute_cost_with_regularization_test_case()
# nn = Neural_network(Y=Y_assess, params=parameters, weight_decay=0.1)
# print("cost = "+str(nn.cost_function(A3)))


"""Backprop with Reg test"""
# X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()
# (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
# single_layer_cache = (((X_assess, W1, b1), Z1), ((A1, W2, b2), Z2), ((A2, W3, b3), Z3))

# nn = Neural_network()
# nn.X = X_assess
# nn.Y = Y_assess
# nn.cache_list = single_layer_cache
# nn.L = 4
# nn.weight_decay = 0.7

# grads =  nn.multi_layer_backward(A3)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("dW3 = "+ str(grads["dW3"]))

"""Model test: Minibatch"""
train_X, train_Y = load_dataset()
# train 3-layer model
layer_dims = [train_X.shape[0], 5, 2, 1]
nn = Neural_network(layer_dims=layer_dims)
nn.X = train_X
nn.Y = train_Y
nn.no_epoches = 10000
nn.learning_rate = 0.0007

parameters = nn.model(plot_learning_curve=True)

# Predict
acc_train = nn.evaluate()
print("Accuracy on training set:", acc_train)

plt.show()