#pragma once

#include "distconv/cudnn/backend.hpp"

#include <Al.hpp>

#include <memory>

namespace distconv {

template <>
class CrossEntropy<cudnn::BackendCUDNN> {
 public:
  CrossEntropy(cudnn::BackendCUDNN &backend, const bool use_labels=false): m_be(backend), m_use_labels(use_labels) {}

  ~CrossEntropy() = default;

  CrossEntropy &operator=(const CrossEntropy &x) {
    assert_always(&m_be == &x.m_be);
    assert_always(m_use_labels == x.m_use_labels);
    return *this;
  }

  template <typename Tensor>
  void setup(const Tensor &x_pred, const Tensor &x_truth,
             const Tensor &y) {
    // Both tensors must have the same process grid
    assert_eq(x_pred.get_locale_shape(),
              x_truth.get_locale_shape());
    if(m_use_labels) {
      // Must have the same number of samples
      assert_eq(x_pred.get_shape()[-1], x_truth.get_shape()[-1]);
      assert_eq(x_pred.get_local_shape()[-1], x_truth.get_local_shape()[-1]);
      // Must have the same global and locale spatial shapes
      for(int i = 0; i < x_pred.get_num_spatial_dims(); i++) {
        assert_eq(x_pred.get_shape()[i], x_truth.get_shape()[i]);
        assert_eq(x_pred.get_local_shape()[i], x_truth.get_local_shape()[i]);
      }
      // Must have only one channel
      assert_eq(x_truth.get_shape()[-2], 1);
      assert_eq(x_truth.get_local_shape()[-2], 1);
    } else {
      // Must have the same global and locale shapes
      assert_eq(x_pred.get_shape(), x_truth.get_shape());
      assert_eq(x_pred.get_local_shape(), x_truth.get_local_shape());
    }
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
  const bool m_use_labels;
  int m_num_procs_per_sample;
  std::unique_ptr<Al::NCCLBackend::comm_type> m_al;
};

} // namespace distconv
