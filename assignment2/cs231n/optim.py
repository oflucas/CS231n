import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
  """
  Performs vanilla stochastic gradient descent.

  config format:
  - learning_rate: Scalar learning rate.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)

  w -= config['learning_rate'] * dw
  return w, config


def sgd_momentum(w, dw, config=None):
  """
  Performs stochastic gradient descent with momentum.

  config format:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    Setting momentum = 0 reduces to sgd.
  - velocity: A numpy array of the same shape as w and dw used to store a moving
    average of the gradients.
  """
  if config is None: config = {}
  alpha = config.get('learning_rate', 1e-2)
  momentum = config.get('momentum', 0.9)
  v = config.get('velocity', np.zeros_like(w))
  
  next_w = None
  #############################################################################
  # TODO: Implement the momentum update formula. Store the updated value in   #
  # the next_w variable. You should also use and update the velocity v.       #
  #############################################################################
  v = momentum * v - alpha * dw
  next_w = w + v
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  config['velocity'] = v

  return next_w, config



def rmsprop(x, dx, config=None):
  """
  Uses the RMSProp update rule, which uses a moving average of squared gradient
  values to set adaptive per-parameter learning rates.
  
  RMSProp is simple as: 
  Tijmen Tieleman and Geoffrey Hinton. "Lecture 6.5-rmsprop: 
  Divide the gradient by a running average of its recent magnitude." 
  COURSERA: Neural Networks for Machine Learning 4 (2012).

  config format:
  - learning_rate: Scalar learning rate.
  - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
    gradient cache.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - cache: Moving average of second moments of gradients.
  """
  if config is None: config = {}
  learning_rate = config.get('learning_rate', 1e-2)
  decay_rate = config.get('decay_rate', 0.99)
  eps = config.get('epsilon', 1e-8)
  cache = config.get('cache', np.zeros_like(x))

  next_x = None
  #############################################################################
  # TODO: Implement the RMSprop update formula, storing the next value of x   #
  # in the next_x variable. Don't forget to update cache value stored in      #  
  # config['cache'].                                                          #
  #############################################################################
  cache = decay_rate * cache + (1 - decay_rate) * dx**2
  next_x = x - learning_rate * dx / (np.sqrt(cache) + eps)
  
  config['cache'] = cache
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return next_x, config


def adam(x, dx, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.

  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
      * velocity's decay rate
  - beta2: Decay rate for moving average of second moment of gradient.
      * factor of moving average of magnitudes
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  learning_rate = config.get('learning_rate', 1e-3)
  beta1 = config.get('beta1', 0.9)
  decay_rate = config.get('beta2', 0.999)
  eps = config.get('epsilon', 1e-8)
  velocity = config.get('m', np.zeros_like(x))
  mag = config.get('v', np.zeros_like(x)) # moving average of magnitudes
  t = config.get('t', 0)
  
  next_x = None
  #############################################################################
  # TODO: Implement the Adam update formula, storing the next value of x in   #
  # the next_x variable. Don't forget to update the m, v, and t variables     #
  # stored in config.                                                         #
  #############################################################################
  t += 1
  velocity = beta1 * velocity + (1 - beta1) * dx
  mag = decay_rate * mag + (1 - decay_rate) * dx**2
  
  velocity_ = velocity / (1 - beta1 ** t) # bias-corrected
  mag_ = mag / (1 - decay_rate ** t) # bias-corrected
    
  next_x = x - learning_rate * velocity_ / (np.sqrt(mag_) + eps)

  config['t'] = t
  config['v'] = mag
  config['m'] = velocity
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return next_x, config

  
  
  

