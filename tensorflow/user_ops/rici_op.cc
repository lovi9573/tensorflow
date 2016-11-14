/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example Op.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

REGISTER_OP("RotationInvariantConvolutionNaive")
    	    .Input("input: T")
    	    .Input("filter: T")
    	    .Output("output: T")
    	    .Attr("T: {float, double}")
    	    .Attr("strides: list(int)")
    	    .Attr(GetPaddingAttrString())
    	    .SetShapeFn(shape_inference::DepthwiseConv2DNativeShape)
    	    .Doc(R"doc(
    	Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

    	Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
    	and a filter / kernel tensor of shape
    	`[filter_height, filter_width, in_channels, channel_multiplier]`, containing
    	`in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
    	a different filter to each input channel (expanding from 1 channel to
    	`channel_multiplier` channels for each), then concatenates the results
    	together. Thus, the output has `in_channels * channel_multiplier` channels.

    	for k in 0..in_channels-1
    	  for q in 0..channel_multiplier-1
    	    output[b, i, j, k * channel_multiplier + q] =
    	      sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
    	                        filter[di, dj, k, q]

    	Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
    	horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

    	strides: 1-D of length 4.  The stride of the sliding window for each dimension
    	  of `input`.
    	padding: The type of padding algorithm to use.
    	)doc");

class RotationInvariantConvolutionNaiveOp : public OpKernel {
 public:
  explicit RotationInvariantConvolutionNaiveOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Output a scalar string.
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &output_tensor));

    auto output = output_tensor->scalar<string>();

    output() = "A(m, 0) == A(m-1, 1)";
  }
};

REGISTER_KERNEL_BUILDER(Name("RotationInvariantConvolutionNaive").Device(DEVICE_CPU), RotationInvariantConvolutionNaiveOp);

}  // namespace tensorflow
