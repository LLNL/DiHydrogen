#pragma once

#include "distconv/cudnn/backend.hpp"

#include <Al.hpp>

#include <memory>

namespace distconv {

template <>
class MeanSquaredError<cudnn::BackendCUDNN> {
 public:
  MeanSquaredError(cudnn::BackendCUDNN &backend): m_be(backend) {}

  ~MeanSquaredError() = default;

  MeanSquaredError &operator=(const MeanSquaredError &x) {
    assert_always(&m_be == &x.m_be);
    return *this;
  }

  template <typename Tensor>
  void setup(const Tensor &x_pred, const Tensor &x_truth,
             const Tensor &y) {
    // Both tensors must have the same process grid
    assert_eq(x_pred.get_locale_shape(),
              x_truth.get_locale_shape());
    // Must have the same global and locale shapes
    assert_eq(x_pred.get_shape(), x_truth.get_shape());
    assert_eq(x_pred.get_local_shape(), x_truth.get_local_shape());
    // No halo for simplicity
    assert_eq(x_pred.get_local_shape(),
              x_pred.get_local_real_shape());
    assert_eq(x_truth.get_local_shape(),
              x_truth.get_local_real_shape());
    // y is a 1-d vector of length as long as the number of samples
    assert_eq(x_pred.get_local_shape()[-1], y.get_local_shape()[-1]);
    assert_eq(y.get_local_shape()[-1], y.get_local_size());
    // same process grid
    assert_eq(x_pred.get_locale_shape(), y.get_locale_shape());
    // no partitioning except for the sample dimension
    assert_eq(y.get_split_shape().reduce_prod(), y.get_split_shape()[-1]);
    assert_eq(y.get_locale_shape()[-1], y.get_split_shape()[-1]);

    auto loc_shape = x_pred.get_locale_shape();
    m_num_procs_per_sample = loc_shape.reduce_sum() / loc_shape[-1];
    if (m_num_procs_per_sample > 1) {
      auto sample_loc = x_pred.get_sub_locale_except_dim(-1);
      m_al.reset(new Al::NCCLBackend::comm_type(
          sample_loc.get_comm(), m_be.get_stream()));
    }
  }

  template <typename Tensor>
  int forward(const Tensor &x_pred, const Tensor &x_truth, Tensor &y);

  template <typename Tensor>
  int backward(const Tensor &x_pred, const Tensor &x_truth,
               Tensor &dy, Tensor &dx_pred,
               Tensor &dx_truth);

 protected:
  cudnn::BackendCUDNN &m_be;
  int m_num_procs_per_sample;
  std::unique_ptr<Al::NCCLBackend::comm_type> m_al;
};

} // namespace distconv
