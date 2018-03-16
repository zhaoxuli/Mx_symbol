# -*- coding: utf-8 -*-

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, act=True):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel,
                              num_group=num_group, stride=stride, pad=pad, no_bias=True)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=True)
    if act:
      act = mx.sym.Activation(data=bn, act_type='relu')
      return act
    else:
      return bn

def GAP_bolck(data, input_num ,output_num, reduce_num)

    conv_dim_redu = Conv(data =data ,num_filter=reduce_num ,kernel=(1, 1),stride = (1,1) , pad=(0,0), act =False)
    conv_out = Conv (data =conv_dim_redu ,num_filter=output_num ,kernel=(1, 1),stride = (1,1) , pad=(0,0))

    #suggest do a max pooling  before do avg pooling if each class can not  output a higher score

    #conv_out = mx.sym.Pooling(data=conv_out, kernel=(2, 2), stride=(1, 1), pool_type="max")
    pool = mx.symbol.Pooling (data = conv_out , global_pool = True , pool_type= 'avg')

    return pool


