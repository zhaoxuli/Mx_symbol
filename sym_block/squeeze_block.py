# -*- coding: utf-8 -*-
#Squeeze_net block  called fire_module
def squeeze(data, num_filter, kernel=(1,1), stride=(1,1), pad=(0, 0), act_type="relu", mirror_attr={}):
	squeeze_1x1=mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
	act=mx.symbol.Activation(data = squeeze_1x1, act_type=act_type, attr=mirror_attr)
	return act

def Fire_module(data, num_filter_squeeze, num_filter_fire,kernel_sequeeze=(1,1),kernel_1x1=(1,1),
                kernel_3x3=(3,3),stride_squeeze=(1,1),stride_1x1=(1,1), stride_3x3=(1,1),
				pad_1x1=(0, 0), pad_3x3=(1, 1),act_type="relu", mirror_attr={}):
	squeeze_1x1=squeeze(data, num_filter_squeeze,kernel_sequeeze,stride_squeeze)
    #squeeze 96 feature maps to 16 by 1*1 conv
	expand1x1=mx.symbol.Convolution(data=squeeze_1x1, num_filter=num_filter_fire, kernel=kernel_1x1, stride=stride_1x1, pad=pad_1x1)
    #expand 16 feature maps to 64 by 1*1conv
	relu_expand1x1=mx.symbol.Activation(data = expand1x1, act_type=act_type, attr=mirror_attr)
    #activation

	expand3x3=mx.symbol.Convolution(data=squeeze_1x1, num_filter=num_filter_fire, kernel=kernel_3x3, stride=stride_3x3, pad=pad_3x3)
    #expand 16 feature maps to 64 by 3*3conv
	relu_expand3x3=mx.symbol.Activation(data = expand3x3, act_type=act_type, attr=mirror_attr)
    #activation

    #in fact the width and height is not changed after squeeze_1x1 and  expand 1*1 and 3*3
    #sum expand1x1 and expand3x3 feature maps  as resulte !!!!  why add?  how about multiply channel' weight???
	return relu_expand1x1+relu_expand3x3

