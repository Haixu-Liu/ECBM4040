import numpy as np
from random import shuffle

def softmax(x):
    h = np.zeros_like(x)
    for i in range(x.shape[0]):
        e_xi = np.exp(x[i] - np.max(x[i]))
        h[i] = e_xi / e_xi.sum()
    return h

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    # construct a one-hot vector for y
    onehot_y = np.zeros((y.size, W.shape[1]))
    onehot_y[np.arange(y.size), y] = 1
    dW = dW.T
    for i in range(y.shape[0]):
        f = np.dot(X[i], W)
    
        for j in range(W.shape[1]):
            e_f = np.exp(f - np.max(f))
            softmax = e_f / e_f.sum()
            loss -= onehot_y[i][j] * np.log(softmax[j])
            dW[j] -= X[i] * (onehot_y[i][j] - softmax[j])
    
    loss = loss / y.shape[0] + reg * np.linalg.norm(W)
    dW = dW.T / y.shape[0] + 2 * reg * W

    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    # construct a one-hot vector for y
    onehot_y = np.zeros((y.size, W.shape[1]))
    onehot_y[np.arange(y.size), y] = 1
    f = np.dot(X, W)
    loss = -np.multiply(onehot_y, np.log(softmax(f)))
    loss = loss.sum() / y.shape[0] + reg * np.linalg.norm(W)
    dW = -np.dot(X.T, (onehot_y - softmax(f))) / y.shape[0]
    dW += 2 * reg * W
    
    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW

# the third way to implement softmax, using TensorFlow, reference to verification code
import tensorflow as tf
def softmax_loss_tf(W, X, y, reg):
    with tf.GradientTape() as tape:
        W_tf = tf.Variable(W, dtype=tf.float32)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=\
                        tf.matmul(X, W_tf), labels=tf.one_hot(y, 10))
        loss_gt = tf.reduce_mean(cross_entropy) + reg * tf.reduce_sum(W_tf * W_tf)
        grad_gt = tape.gradient(loss_gt, W_tf)
        
        return loss_gt, grad_gt
