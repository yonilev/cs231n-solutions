import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    f = X[i].dot(W)
    f -= np.max(f)
    sum_f = np.sum(np.exp(f))
    loss += -f[y[i]] + np.log(sum_f)

    p = np.exp(f)/sum_f
    for c in range(num_classes):
        if c==y[i]:
            dW[:,c] += p[c]*X[i] - X[i]
        else:
            dW[:,c] += p[c]*X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  dW += W * reg

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)



  return loss, dW




def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]

  f = X.dot(W)
  f -= np.max(f,axis=1).reshape(num_train,1)
  sum_f = np.sum(np.exp(f),axis=1).reshape(num_train,1)

  loss = np.sum(-f[np.arange(num_train),y].reshape(num_train,1) + np.log(sum_f))

  p = np.exp(f)/sum_f
  p[np.arange(num_train),y] -= 1
  dW = X.T.dot(p)

  loss /= num_train
  dW /= num_train
  dW += W * reg

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)



  return loss, dW

