#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-09-27 19:24
import sys

sys.path.insert(0, "/home/sunway/source/apollo_flops/caffe/python")
import caffe
import sys

caffe.set_mode_cpu()
import numpy as np
from numpy import prod, sum
from pprint import pprint


def print_flops(deploy_file):
    print("Net: " + deploy_file)
    net = caffe.Net(deploy_file, caffe.TEST)
    flops = 0
    dict = {a: b for a, b in zip(net._layer_names, [x.type for x in net.layers])}

    for layer_name, blob in net.blobs.items():
        if layer_name not in dict:
            continue
        cur_flops = 0.0
        if dict[layer_name] in ("Convolution", "Deconvolution"):
            # OIHW
            kernel_shape = net.params[layer_name][0].data.shape
            # NCHW
            output_shape = blob.data.shape
            cur_flops = np.product(kernel_shape[1:]) * np.product(output_shape) * 2
            print(
                f"{layer_name:<20s} {dict[layer_name]:<20s} {cur_flops/1024/1024:<20.0f} {str(kernel_shape):<20s} {str(output_shape):<20s}"
            )

        if dict[layer_name] == "DepthwiseConvolution":
            print("a")
            # OIHW
            kernel_shape = net.params[layer_name][0].data.shape
            # NCHW
            output_shape = blob.data.shape
            # depthwise conv
            cur_flops += np.prod(kernel_shape[2:]) * np.prod(output_shape) * 2
            # 1x1 conv
            cur_flops += kernel_shape[1] * np.prod(output_shape) * 2
            print(
                f"{layer_name:<20s} {dict[layer_name]:<20s} {cur_flops/1024/1024:<20.0f} M {str(kernel_shape):<20s} {str(output_shape):<20s}"
            )

        if dict[layer_name] == "InnerProduct":
            weight_shape = net.params[layer_name][0].data.shape
            cur_flops += np.prod(weight_shape) * 2
            print(
                f"{layer_name:<20s} {dict[layer_name]:<20s} {cur_flops/1024/1024:<20.0f} M {str(weight_shape):<20s} "
            )
        flops += cur_flops

    print("layers num: " + str(len(net.params.items())))
    print(f"Total number of flops: {flops / (1024 * 1024):.1f} M")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("python caffe_flops.py deploy.prototxt")
        exit()
    deploy_file = sys.argv[1]
    print_flops(deploy_file)
