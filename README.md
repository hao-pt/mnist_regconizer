# mnist_regconizer
This is a first lab of Soft computing course. It aims to implement neural network (1 hidden layer) and optimize this network to increase the accuracy of the model. However, I've generalized the network for L layers in implementation.

## Install packages
```
pip install -r requirements.txt
```

## Model architecture
Neural network includes 1 or multiple hidden layers but here I just experiments with 1 or 2 hidden layers.

## Implementation specifications:
```
Input: (784, m) - m is number of examples

Output: (1, m)
```
> Main library is numpy.

- Weight initialization is `He initialization` to avoid vanishing/exploding gradient at starting epochs and work well with Relu activation function.
- Optimizer: minibatch gradient descent.
- Regularization: weight decay (L2 norm) to avoid overfitting.
- Loss function: softmax cross-entropy for multivariate problem.

Note: Activation function Relu is used at hiden layers while Softmax is put at output layer.

## Tuning parameters

Some hyper-parameters to tune:
- nx: dimension of X
- nh1: number of units in hidden layer 1 số units của hidden layer 1
- nh2: number of units in hidden layer 2 số units của hidden layer 2
- epoches: number of epochs to train model số epoch để train model
- batch size
- learning rate
- weight decay: L2 regularization

## Run commandline
Get help:

```
python 1612174.py -h
```
 
Train model:

```
python 1612174.py -train "Your train file" -test digit-recognizer/test.csv -nh1 1000 -epoches 50 -batch 64 -lr 0.5 -decay 5e-4
```

Parameters are saved in filename: ddmmYY-HMS (dd/mm/Y H:M:S)

Ex: 08102019-165350
