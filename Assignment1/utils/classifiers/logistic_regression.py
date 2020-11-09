import numpy as np
from random import shuffle

def sigmoid(x):
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                                         #         
    #############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    h = 1 / (1 + np.exp(-x))
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    
    
    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    dW = dW.T
    # construct a one-hot vector for y
    onehot_y = np.zeros((y.size, W.shape[1]))
    onehot_y[np.arange(y.size), y] = 1
    # iterate through all the data sample to calculate loss and gradient
    for i in range(y.shape[0]):
        # f is the prob score of datasample X[i], of shape 1 * 2
        f = sigmoid(np.dot(X[i], W)).T
        
        for j in range(W.shape[1]):
            loss += -(onehot_y[i][j] * np.log(f[j]) + (1 - onehot_y[i][j]) * np.log(1 - f[j]))
            dW[j] += -(onehot_y[i][j] - f[j]) * X[i]
    
    loss = loss / y.shape[0] + reg * np.linalg.norm(W)
    dW = dW.T / y.shape[0] + 2 * reg * W
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################
    

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no     # 
    # explicit loops.                                                          #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    ############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    # construct a one-hot vector for y
    onehot_y = np.zeros((y.size, W.shape[1]))
    onehot_y[np.arange(y.size), y] = 1
    ones = np.ones((onehot_y.shape))
    z = np.dot(X, W)
    h = sigmoid(z)
    loss = -(np.multiply(onehot_y, np.log(h)) + np.multiply((ones - onehot_y),np.log(ones - h)))
    loss = loss.sum() / y.shape[0]
    loss += reg * np.linalg.norm(W)
    dW = -np.dot(X.T, (onehot_y - h)) / y.shape[0]
    dW += 2 * reg * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    

    return loss, dW
