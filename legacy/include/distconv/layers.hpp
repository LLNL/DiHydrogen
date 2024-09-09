#pragma once

#include "distconv/base.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"

namespace distconv
{

template <typename Backend, typename DataType>
class Convolution;

template <typename Backend, typename DataType>
class Pooling;

template <typename Backend>
class ReLU;

template <typename Backend>
class LeakyReLU;

template <typename Backend, typename DataType>
class BatchNormalization;

enum class SoftmaxMode
{
  INSTANCE,
  CHANNEL
};

template <typename Backend>
class Softmax;

template <typename Backend>
class CrossEntropy;

template <typename Backend>
class MeanSquaredError;

}  // namespace distconv
