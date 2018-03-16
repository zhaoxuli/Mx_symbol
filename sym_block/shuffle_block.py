# -*- coding: utf-8 -*-

def combine(residual, data, combine):
	if combine == 'add':
		return residual + data
	elif combine == 'concat':
		return mx.sym.concat(residual, data, dim=1)
	return None

def channel_shuffle(data, groups):
	data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
	data = mx.sym.swapaxes(data, 1, 2)
	data = mx.sym.reshape(data, shape=(0, -3, -2))
	return data

def shuffleUnit(residual, in_channels, out_channels, combine_type, groups=3, grouped_conv=True):

    # there are 2 block formats in paper add and concat

    if combine_type == 'add':
        DWConv_stride = 1
    elif combine_type == 'concat':
        DWConv_stride = 2
        out_channels -= in_channels

    first_groups = groups if grouped_conv else 1

    bottleneck_channels = out_channels // 4  #floor

    data = mx.sym.Convolution(data=residual, num_filter=bottleneck_channels,
    	              kernel=(1, 1), stride=(1, 1), num_group=first_groups)
    data = mx.sym.BatchNorm(data=data)
    data = mx.sym.Activation(data=data, act_type='relu')

    data = channel_shuffle(data, groups)

    data = mx.sym.Convolution(data=data, num_filter=bottleneck_channels, kernel=(3, 3),
    	               pad=(1, 1), stride=(DWConv_stride, DWConv_stride), num_group=groups)
    data = mx.sym.BatchNorm(data=data)

    data = mx.sym.Convolution(data=data, num_filter=out_channels,
    	               kernel=(1, 1), stride=(1, 1), num_group=groups)
    data = mx.sym.BatchNorm(data=data)

    if combine_type == 'concat':
        residual = mx.sym.Pooling(data=residual, kernel=(3, 3), pool_type='avg',
        	                  stride=(2, 2), pad=(1, 1))

    data = combine(residual, data, combine_type)

    return data

