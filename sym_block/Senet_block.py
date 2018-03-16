# -*- coding: utf-8 -*-

def se_unit(data,name,kernel_size,num_filter,ratio=0.25):
    # do spatial_reduce  get num_filter's  number
    squeeze = mx.sym.Pooling(data=data, kernel=kernel_size, pool_type= 'avg',global_pool =True, name=name+'_pool')
    squeeze = mx.symbol.Flatten(data=squeeze ,name =name+'_flatten' )
    #get each channel weight by FullyConnected , the scale control the acc ratio
    excitation = mx.symbol.FullyConnected(data=squeeze, num_hidden=int(num_filter*ratio), name=name+'_fc')
    excitation = mx.sym.Activation(data=excitation, act_type='relu', name=name+'relu')
    excitation = mx.symbol.FullyConnected(data=excitation,num_hidden=num_filter, name=name+'_fc1')
    excitation = mx.sym.Activation(data=excitation, act_type='sigmoid', name=name+'sigmoid')
    #multi
    conv = mx.symbol.broadcast_mul(data,mx.symbol.reshape(data=excitation,shape=(-1,num_filter,1,1)))
    return conv
