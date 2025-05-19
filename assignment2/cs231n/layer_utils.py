from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    前向传播：Affine -> BatchNorm -> ReLU

    Inputs:
    - x: 输入数据，形状 (N, d_1, ..., d_k)
    - w: 权重矩阵，形状 (D, H)
    - b: 偏置向量，形状 (H,)
    - gamma: BN缩放参数，形状 (H,)
    - beta: BN平移参数，形状 (H,)
    - bn_param: BN配置参数字典

    Returns:
    - out: 输出结果，形状 (N, H)
    - cache: 缓存用于反向传播的元组
    """
    a, fc_cache = affine_forward(x, w, b)
    an, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache):
    """
    反向传播：ReLU -> BatchNorm -> Affine

    Input:
    - dout: 上游梯度，形状 (N, H)
    - cache: 缓存的元组

    Returns:
    - dx: 输入x的梯度，形状 (N, d_1, ..., d_k)
    - dw: 权重w的梯度，形状 (D, H)
    - db: 偏置b的梯度，形状 (H,)
    - dgamma: gamma的梯度，形状 (H,)
    - dbeta: beta的梯度，形状 (H,)
    """
    fc_cache, bn_cache, relu_cache = cache

    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward_alt(dan, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)

    return dx, dw, db, dgamma, dbeta

def affine_ln_relu_forward(x, w, b, gamma, beta, ln_param):
    """
    前向传播：Affine -> LayerNorm -> ReLU

    Inputs:
    - x: 输入数据，形状 (N, D)
    - w: 权重矩阵，形状 (D, H)
    - b: 偏置向量，形状 (H,)
    - gamma: 缩放参数，形状 (H,)
    - beta: 平移参数，形状 (H,)
    - ln_param: LayerNorm配置参数字典

    Returns:
    - out: 输出数据，形状 (N, H)
    - cache: 反向传播所需的缓存
    """
    a, fc_cache = affine_forward(x, w, b)                 # Affine层输出 (N, H)
    ln_out, ln_cache = layernorm_forward(a, gamma, beta, ln_param)  # LayerNorm输出 (N, H)
    out, relu_cache = relu_forward(ln_out)                # ReLU输出 (N, H)
    cache = (fc_cache, ln_cache, relu_cache)
    return out, cache

def affine_ln_relu_backward(dout, cache):
    """
    反向传播：ReLU -> LayerNorm -> Affine

    Inputs:
    - dout: 上游梯度，形状 (N, H)
    - cache: 前向传播缓存的元组

    Returns:
    - dx: 输入x的梯度，形状 (N, D)
    - dw: 权重w的梯度，形状 (D, H)
    - db: 偏置b的梯度，形状 (H,)
    - dgamma: gamma的梯度，形状 (H,)
    - dbeta: beta的梯度，形状 (H,)
    """
    fc_cache, ln_cache, relu_cache = cache

    dln = relu_backward(dout, relu_cache)                      # ReLU反向传播
    da, dgamma, dbeta = layernorm_backward(dln, ln_cache)     # LayerNorm反向传播
    dx, dw, db = affine_backward(da, fc_cache)                 # Affine反向传播

    return dx, dw, db, dgamma, dbeta

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
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
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
