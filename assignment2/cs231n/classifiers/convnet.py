import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

from cs231n.classifiers.fc_net import *

class MyConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - FullyConnectedNet
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim, fc_hidden_dims, num_classes, 
               num_filters=32, filter_size=7, weight_scale=1e-3, reg=0.0,
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
    
    # My Changes
    self.F, self.WW = num_filters, filter_size
    self.C, self.H, self.W = input_dim
    
    # pass conv_param to the forward pass for the convolutional layer
    self.conv_param = {'stride': 1, 'pad': (self.WW - 1) / 2}
    
    # pass pool_param to the forward pass for the max-pooling layer
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    self.params['W1'] = weight_scale * np.random.randn(self.F, self.C, self.WW, self.WW)
    self.params['b1'] = weight_scale * np.zeros(self.F)
    # Input = (F, H/2, W/2), will be flatened in affine layer, Output = (hidden_dim,)
    self.fc = FullyConnectedNet(fc_hidden_dims, self.F * self.H/2 * self.W/2, num_classes=num_classes, 
                                dropout=dropout, use_batchnorm=use_batchnorm, reg=reg, weight_scale=weight_scale, 
                                dtype=dtype, seed=seed)
    for p, val in self.fc.params.items():
      self.params['fc:'+p] = val

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    for p in self.fc.params.keys():
      self.fc.params[p] = self.params['fc:'+p]


    X, cache1 = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
    
    if y is None:
      scores = self.fc.loss(X)
      return scores
    
    grads = {}
    loss, fc_grads = self.fc.loss(X, y)
    _,  grads['W1'], grads['b1'] = conv_relu_pool_backward(fc_grads['x'], cache1)    
    
    for w in ['W1']:
      loss += 0.5 * self.reg * np.sum(self.params[w]**2)
      grads[w] += self.reg * self.params[w]
        
    for p in self.fc.params.keys():
      grads['fc:'+p] = fc_grads[p]
    
    return loss, grads
