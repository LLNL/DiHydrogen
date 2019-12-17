#!/usr/bin/env python3

"""
Perofrm multi-dimensional max-pooling to a tensor stored in `PATH_INPUT`,
and then output the result to `PATH_OUTPUT`.
This script accepts arguments equivalent to `cudnn_benchmark`.

This script requires the Chainer framework, which can be installed via
`pip install chainer`. See the following website for more details:
https://docs.chainer.org/en/stable/install.html
"""

import argparse

import numpy as np
import cupy as cp

import chainer
from chainer.functions.pooling.max_pooling_nd import MaxPoolingND

PATH_INPUT  = "input_tensor_ref.txt"
PATH_OUTPUT = "output_tensor.txt"

# TODO: Support d_*_tensor.txt (back-propagation).

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="A reference implementation of ND-pooling.")
   for a in ["num-dims", "num-channels", "num-samples", "num-filters",
             "depth", "height", "width",
             "filter-depth", "filter-height", "filter-width"]:
      parser.add_argument("--{}".format(a),  type=int)

   parser.add_argument("--dump-input", dest="dump_input",
                       action="store_const", const=True, default=False)
   parser.add_argument("--dump-output", dest="dump_output",
                       action="store_const", const=True, default=False)
   parser.add_argument("--cpu", dest="cpu",
                       action="store_const", const=True, default=False)

   args = parser.parse_args()
   num_dims = args.num_dims
   assert num_dims == 2 or num_dims == 3
   spatial_args = (["depth"] if num_dims == 3 else []) + ["height", "width"]
   image_size = [args.num_samples, args.num_channels] + \
      [getattr(args, x) for x in spatial_args]
   filter_size = [getattr(args, "filter_{}".format(x)) for x in spatial_args]
   assert num_dims == len(image_size)-2
   assert num_dims == len(filter_size)

   tensor = np.loadtxt(PATH_INPUT)
   assert tensor.shape == (np.prod(image_size), )
   x = tensor.reshape(list(image_size))

   pad = list(((np.array(filter_size)-1)/2).astype(np.int))
   stride = [1]*num_dims

   pooling = MaxPoolingND(
      ndim=num_dims,
      ksize=filter_size,
      stride=stride,
      pad=pad,
      cover_all=False)

   if not args.cpu:
      x = cp.array(x)

   y, = pooling.forward((x, ))

   if not args.cpu:
      y = cp.asnumpy(y)

   assert x.shape == y.shape
   np.savetxt(PATH_OUTPUT, y.reshape(-1).astype(np.float32))
