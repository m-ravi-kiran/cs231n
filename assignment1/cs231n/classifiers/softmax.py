import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    # calculate scores of each example
    score = X[i].dot(W)
    # normalization to exp function not to explode
    score -= np.max(score)
    num = 0
    den = np.sum(np.exp(score))
    for j in range(num_classes):
      if j == y[i]:
        num += np.exp(score[j]) # correct class score
    loss += - np.log(num/den)
    for j in range(num_classes):
      num = np.exp(score[j])
      dW[:,j] += ((num/den) - (j == y[i])) * X[i]# (D,C)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
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
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)
  exp_scores = np.exp(scores)
  sums = np.sum(exp_scores, axis=1, keepdims=True)
  scores = exp_scores/ sums
  # range(num_train) is used below to traverse through all the lines and all correct class positions
  loss = np.sum(-np.log(scores[range(num_train), y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  ind = np.zeros_like(scores)
  ind[range(num_train),y] = 1
  dW = X.T.dot(scores-ind)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
