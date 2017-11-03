import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  margin_arr = np.zeros((num_train, num_classes))
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W) # 1xD * DxC = 1xC
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        margin_arr[i, j] = margin
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # ans find in: http://cs231n.github.io/optimization-1/#gradcompute
  for i in xrange(num_train):
    for j in xrange(num_classes):
      if j != y[i] and margin_arr[i, j] > 0.:
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]
  dW /= num_train
  dW += reg * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  margins = np.zeros((num_train, num_classes))
  scores = X.dot(W) # NxD * DxC = NxC
  each_row = np.arange(num_train)
  correct_scores = scores[each_row, y] # (N,)
  margins = np.maximum(0, scores - correct_scores[:,np.newaxis] + 1.)
  margins[each_row, y] = 0
  loss = np.sum(margins)
  loss /= num_train # avg by batch size
  loss += 0.5 * reg * np.sum(W * W) # Add regularization to the loss.
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # half vectorized version
  #for i in xrange(num_train):
  #  musk = margins[i] > 0.
  #  dW.T[musk] += X[i]
  #  dW[:,y[i]] -= sum(musk) * X[i]
  
  # https://github.com/huyouare/CS231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py
  m = np.zeros(margins.shape)
  m[margins > 0] = 1
  m[np.arange(num_train), y] = -1 * np.sum(m, axis=1)
  dW = np.dot(X.T, m)
    
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

def svm_predict(W, X):
  """
  X is (N, D)
  W is (D, C)
  return y
  """
  scores = X.dot(W) # (N,C)
  y = np.argmax(scores, axis=1) # get argmax on each row (axis=1)
  return y