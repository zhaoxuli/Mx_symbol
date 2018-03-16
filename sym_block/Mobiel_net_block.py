# -*- coding: utf-8 -*-
def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None):
      conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel,
                                num_group=num_group, stride=stride, pad=pad, no_bias=True, name= name)
      bn = mx.sym.BatchNorm(data=conv, name= name, fix_gamma=True)
      act = mx.sym.Activation(data=bn, act_type='relu', name= name)
      return act


def  Mobiel_bolck(data,input_num,output_num,kernel_size,pad_size):
    conv_dw = Conv(data, num_group=input_num, num_filter=input_num,
                   kernel=kernel_size, pad=pad_size, stride=(1, 1))
    conv_pw = Conv(conv_dw, num_filter= output_num, kernel=(1, 1),
                   pad=(0, 0), stride=(1, 1))
    return conv_pw

