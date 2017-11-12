import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

from cs231n.classifiers.fc_net import *


def conv_bn_relu_pool_forward(x, w, b, gamma, beta, conv_param, pool_param, bn_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  a, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache

def conv_bn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta

class MyConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  [conv - batch norm - relu - 2x2 max pool] * N - FullyConnectedNet
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim, fc_hidden_dims, num_classes, 
               num_filters=[32], filter_size=7, weight_scale=1e-3, reg=0.0,
               dropout=0., use_batchnorm=False,
               dtype=np.float32, seed=None):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_conv_layers = len(num_filters)
    
    self.F, self.WW = num_filters, filter_size
    self.C, self.H, self.W = input_dim
    
    # pass conv_param to the forward pass for the convolutional layer
    self.conv_param = {'stride': 1, 'pad': (self.WW - 1) / 2}
    
    # pass pool_param to the forward pass for the max-pooling layer
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    
    for l in xrange(self.num_conv_layers):
      input_channel_num = self.F[l - 1] if l > 0 else self.C
      w_, b_ = w_b(l)
      self.params[w_] = weight_scale * np.random.randn(self.F[l], input_channel_num, self.WW, self.WW)
      self.params[b_] = weight_scale * np.zeros(self.F[l])
        
      gamma_, beta_ = get_gamma_beta_key(l)
      self.params[gamma_] = np.ones(self.F[l])
      self.params[beta_] = np.zeros(self.F[l])
        
    # Input = (F, H/2, W/2), will be flatened in affine layer
    input_d = self.F[-1] * self.H/(2**self.num_conv_layers) * self.W/(2**self.num_conv_layers)
    self.fc = FullyConnectedNet(fc_hidden_dims, input_d, num_classes=num_classes, 
                                dropout=dropout, use_batchnorm=use_batchnorm, reg=reg, weight_scale=weight_scale, 
                                dtype=dtype, seed=seed)
    for p, val in self.fc.params.items():
      self.params['fc:'+p] = val
    
    self.bn_params = [{'mode': 'train'} for i in xrange(self.num_conv_layers)]

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'
 
    for bn_param in self.bn_params:
      bn_param[mode] = mode
    
    for p in self.fc.params.keys():
      self.fc.params[p] = self.params['fc:'+p]

    ############################################################################
    #                             FORWARD PASS                                 #
    ############################################################################     
    caches = []
    for l in xrange(self.num_conv_layers):
      w_, b_ = w_b(l)
      gamma_, beta_ = get_gamma_beta_key(l)
      X, cache_ = conv_bn_relu_pool_forward(X, self.params[w_], self.params[b_], self.params[gamma_], self.params[beta_], 
                                            self.conv_param, self.pool_param, self.bn_params[l])
      caches.append(cache_)
    
    if y is None:
      scores = self.fc.loss(X)
      return scores
    
    ############################################################################
    #                             BACK PROPAGATION                            #
    ############################################################################    
    grads = {}
    loss, fc_grads = self.fc.loss(X, y)
    
    dx = fc_grads['x']
    for l in reversed(range(self.num_conv_layers)):
      w_, b_ = w_b(l)
      gamma_, beta_ = get_gamma_beta_key(l)
      dx,  grads[w_], grads[b_], grads[gamma_], grads[beta_] = conv_bn_relu_pool_backward(dx, caches[l])    
    
      loss += 0.5 * self.reg * np.sum(self.params[w_]**2)
      grads[w_] += self.reg * self.params[w_]
        
    for p in self.fc.params.keys():
      grads['fc:'+p] = fc_grads[p]
    
    return loss, grads
