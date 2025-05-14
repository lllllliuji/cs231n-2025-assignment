from builtins import range
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

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        for j in range(num_classes):
          if j == y[i]:
            dW[:, j] += (p[j] - 1) * X[i]
          else:
            dW[:, j] += p[j] * X[i]

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    dW /= num_train
    dW += 2 * reg * W 
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
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    # Step 1: Compute all scores
    scores = X.dot(W)                      # (N, C)

    # Step 2: Numeric stability: subtract max per row
    scores -= np.max(scores, axis=1, keepdims=True)

    # Step 3: Compute softmax probabilities
    exp_scores = np.exp(scores)            # (N, C)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N, C)

    # Step 4: Compute the loss: average cross-entropy loss
    N = X.shape[0]
    correct_log_probs = -np.log(probs[np.arange(N), y])
    data_loss = np.sum(correct_log_probs) / N
    reg_loss = reg * np.sum(W * W)
    loss = data_loss + reg_loss


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # Step 5: Compute the gradient on scores
    dscores = probs.copy()                 # (N, C)
    dscores[np.arange(N), y] -= 1          # subtract 1 from the correct class

    # Step 6: Backpropagate the gradient to W
    dW = X.T.dot(dscores)                  # (D, N) x (N, C) → (D, C)
    dW /= N
    dW += 2 * reg * W                      # add regularization gradient

    return loss, dW
