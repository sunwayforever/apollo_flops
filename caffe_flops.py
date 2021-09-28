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


def print_net_parameters_flops(deploy_file):
    print("Net: " + deploy_file)
    net = caffe.Net(deploy_file, caffe.TEST)
    flops = 0
    typenames = ["Convolution", "DepthwiseConvolution", "InnerProduct"]

    print("Layer-wise parameters: ")
    print(
        "layer name".ljust(20),
        "Filter Shape".ljust(20),
        "Output Size".ljust(20),
        "Layer Type".ljust(20),
        "Flops".ljust(20),
    )

    dict = {a: b for a, b in zip(net._layer_names, [x.type for x in net.layers])}

    for layer_name, blob in net.blobs.items():
        if layer_name not in dict:
            continue
        if dict[layer_name] in typenames:
            cur_flops = 0.0
            # OIHW
            kernel_shape = net.params[layer_name][0].data.shape
            # NCHW
            output_shape = blob.data.shape
            if dict[layer_name] in typenames[:2]:
                cur_flops = np.product(kernel_shape) * np.product(output_shape) * 2
            else:
                cur_flops = np.product(kernel_shape)
            print(
                layer_name.ljust(20),
                str(net.params[layer_name][0].data.shape).ljust(20),
                str(blob.data.shape).ljust(20),
                dict[layer_name].ljust(20),
                str(cur_flops).ljust(20),
            )
            # InnerProduct
            if len(blob.data.shape) == 2:
                flops += prod(net.params[layer_name][0].data.shape)
            else:
                flops += (
                    prod(net.params[layer_name][0].data.shape)
                    * blob.data.shape[2]
                    * blob.data.shape[3]
                )

    print("layers num: " + str(len(net.params.items())))
    print(
        f"Total number of parameters: {sum([prod(v[0].data.shape) for k, v in net.params.items()]) / (1024 * 1024):.1f} M"
    )
    print(f"Total number of flops: {flops / (1024 * 1024):.1f} M")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("python calc_params.py  deploy.prototxt")
        exit()
    deploy_file = sys.argv[1]
    print_net_parameters_flops(deploy_file)
