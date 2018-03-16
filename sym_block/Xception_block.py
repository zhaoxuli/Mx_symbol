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


def  Xception_bolck(data,input_num,output_num,kernel_size=(3,3),pad_size=(1,1)):
    res_conv = Conv(data,num_filter= output_num,kernel=(1,1),stride=(2,2))

    conv_dw_1 = Conv(data, num_group=input_num, num_filter=input_num,
                   kernel=kernel_size, pad=pad_size, stride=(1, 1),act=False)
    conv_pw_1 = Conv(conv_dw_1, num_filter= output_num, kernel=(1, 1),
                   pad=(0, 0), stride=(1, 1))

    conv_dw_2 = Conv(conv_pw_1, num_group=input_num, num_filter=input_num,
                   kernel=kernel_size, pad=pad_size, stride=(1, 1),act=False)
    conv_pw_2 = Conv(conv_dw_2, num_filter= output_num, kernel=(1, 1),
                     pad=(0, 0), stride=(1, 1))
    pool = mx.sym.Pooling(data=conv_3, kernel=(3, 3), stride=(2, 2), pool_type="max")

    return pool+ res_conv

