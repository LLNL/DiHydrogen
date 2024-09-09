#pragma once

#include "distconv/base.hpp"
#include "distconv/layers.hpp"

namespace distconv
{
namespace ref
{

class Backend
{
public:
  Backend() {}
  std::string get_name() const { return std::string("Ref"); }
};

template <typename Tensor>
void apply4d(typename Tensor::data_type alpha,
             const Tensor& x,
             index_t x_n,
             index_t x_c,
             const Tensor& filter,
             index_t f_k,
             index_t f_c,
             bool rotate,
             typename Tensor::data_type beta,
             Tensor& y,
             index_t y_n,
             index_t y_k,
             int_vector paddings, // DWH
             int_vector strides,  // DWH
             bool expand_halo)
{
  using Array4 = tensor::Array<4>;
  using DataType = typename Tensor::data_type;
  auto shape = expand_halo ? x.get_local_real_shape() : x.get_local_shape();
  index_t h_len = shape[1];
  index_t w_len = shape[0];
  auto f_shape = filter.get_local_shape();
  index_t fh_len = f_shape[1];
  index_t fw_len = f_shape[0];
  const int padding_h = paddings[1];
  const int padding_w = paddings[0];
  index_t h_sweep_len = h_len + padding_h * 2 - fh_len + 1;
  index_t w_sweep_len = w_len + padding_w * 2 - fw_len + 1;
  for (index_t h = 0; h < h_sweep_len; ++h)
  {
    for (index_t w = 0; w < w_sweep_len; ++w)
    {
      DataType acc = 0.0;
      for (index_t i = 0; i < fh_len; ++i)
      {
        for (index_t j = 0; j < fw_len; ++j)
        {
          Array4 x_idx = {w + j, h + i, x_c, x_n};
#if 0
          util::PrintStreamDebug()
              << "idx: " << idx << "\n";
#endif
          DataType xi;
          // if idx is in the padded area, value of 0 is used
          if (x_idx[1] < (index_t) padding_h || x_idx[1] >= h_len + padding_h
              || x_idx[0] < (index_t) padding_w
              || x_idx[0] >= w_len + padding_w)
          {
            xi = 0.0;
          }
          else
          {
            x_idx[0] -= padding_w;
            x_idx[1] -= padding_h;
            xi = x.get(x_idx.get_vector(), expand_halo);
          }
          index_t f_w = rotate ? fw_len - j - 1 : j;
          index_t f_h = rotate ? fh_len - i - 1 : i;
          IndexVector f_idx({f_w, f_h, f_c, f_k});
          DataType fi = filter.get(f_idx);
          acc = acc + xi * fi;
        }
      }
      // save acc to the output tensor
      IndexVector y_idx({w, h, y_k, y_n});
      acc = acc * alpha + y.get(y_idx) * beta;
#if 0
      util::PrintStreamDebug()
          << "output at " << y_idx << ": "
          << acc << "\n";
#endif
      y.set(y_idx, acc);
    }
  }
}

template <typename Tensor>
void apply5d(typename Tensor::data_type alpha,
             const Tensor& x,
             index_t x_n,
             index_t x_c,
             const Tensor& filter,
             index_t f_k,
             index_t f_c,
             bool rotate,
             typename Tensor::data_type beta,
             Tensor& y,
             index_t y_n,
             index_t y_k,
             int_vector paddings, // DWH
             int_vector strides,  // DWH
             bool expand_halo)
{
  using Array5 = tensor::Array<5>;
  using DataType = typename Tensor::data_type;
  auto shape = expand_halo ? x.get_local_real_shape() : x.get_local_shape();
  index_t h_len = shape[2];
  index_t w_len = shape[1];
  index_t d_len = shape[0];
  auto f_shape = filter.get_local_shape();
  index_t fh_len = f_shape[2];
  index_t fw_len = f_shape[1];
  index_t fd_len = f_shape[0];
  const int padding_h = paddings[2];
  const int padding_w = paddings[1];
  const int padding_d = paddings[0];
  index_t h_sweep_len = h_len + padding_h * 2 - fh_len + 1;
  index_t w_sweep_len = w_len + padding_w * 2 - fw_len + 1;
  index_t d_sweep_len = d_len + padding_d * 2 - fd_len + 1;

  for (index_t h = 0; h < h_sweep_len; ++h)
  {
    for (index_t w = 0; w < w_sweep_len; ++w)
    {
      for (index_t d = 0; d < d_sweep_len; ++d)
      {
        DataType acc = 0.0;
        for (index_t i = 0; i < fh_len; ++i)
        {
          for (index_t j = 0; j < fw_len; ++j)
          {
            for (index_t k = 0; k < fd_len; ++k)
            {
              Array5 x_idx = {d + k, w + j, h + i, x_c, x_n};
#if 0
              util::PrintStreamDebug()
                  << "idx: " << idx << "\n";
#endif
              DataType xi;
              // if idx is in the padded area, value of 0 is used
              if (x_idx[2] < (index_t) padding_h
                  || x_idx[2] >= h_len + padding_h
                  || x_idx[1] < (index_t) padding_w
                  || x_idx[1] >= w_len + padding_w
                  || x_idx[0] < (index_t) padding_d
                  || x_idx[0] >= d_len + padding_d)
              {
                xi = 0.0;
              }
              else
              {
                x_idx[0] -= padding_d;
                x_idx[1] -= padding_w;
                x_idx[2] -= padding_h;
                xi = x.get(x_idx.get_vector(), expand_halo);
              }
              index_t f_d = rotate ? fd_len - k - 1 : k;
              index_t f_w = rotate ? fw_len - j - 1 : j;
              index_t f_h = rotate ? fh_len - i - 1 : i;
              IndexVector f_idx({f_d, f_w, f_h, f_c, f_k});
              DataType fi = filter.get(f_idx);
              acc = acc + xi * fi;
            }
          }
        }
        // save acc to the output tensor
        IndexVector y_idx({d, w, h, y_k, y_n});
        acc = acc * alpha + y.get(y_idx) * beta;
#if 0
        util::PrintStreamDebug()
            << "output at " << y_idx << ": "
            << acc << "\n";
#endif
        y.set(y_idx, acc);
      }
    }
  }
}

template <typename Tensor>
void apply(typename Tensor::data_type alpha,
           const Tensor& x,
           index_t x_n,
           index_t x_c,
           const Tensor& filter,
           index_t f_k,
           index_t f_c,
           bool rotate,
           typename Tensor::data_type beta,
           Tensor& y,
           index_t y_n,
           index_t y_k,
           int_vector paddings, // DWH
           int_vector strides,  // DWH
           bool expand_halo)
{
  switch (x.get_num_dims())
  {
  case 4:
    apply4d(alpha,
            x,
            x_n,
            x_c,
            filter,
            f_k,
            f_c,
            rotate,
            beta,
            y,
            y_n,
            y_k,
            paddings,
            strides,
            expand_halo);
    break;
  case 5:
    apply5d(alpha,
            x,
            x_n,
            x_c,
            filter,
            f_k,
            f_c,
            rotate,
            beta,
            y,
            y_n,
            y_k,
            paddings,
            strides,
            expand_halo);
    break;
  default:
    util::PrintStreamError()
      << "Invalid tensor dimension: " << x.get_num_dims();
    std::abort();
  }
}

} // namespace ref

template <typename DataType>
class Convolution<ref::Backend, DataType>
{
public:
  Convolution(ref::Backend& be,
              int num_dims,
              HaloExchangeMethod m = HaloExchangeMethod::MPI,
              bool overlap_halo_exchange = false,
              bool enable_profiling = false)
    : m_be(be), m_num_dims(num_dims)
  {}

  template <typename Tensor>
  void setup(const Tensor& input,
             const Tensor& filter,
             const Tensor& output,
             const Tensor& d_input,
             const Tensor& d_filter,
             const Tensor& d_output,
             const int_vector& pads,
             const int_vector& strides,
             const int_vector dilations,
             int num_groups,
             const std::string& fwd_algo,
             const std::string& bwd_data_algo,
             const std::string& bwd_filter_algo,
             size_t ws_size)
  {
    m_strides = strides;
  }

  template <typename Tensor>
  void setup_bias(const Tensor& bias)
  {
    return;
  }

  template <typename Tensor>
  void setup_bias_gradient(const Tensor& bias_gradient)
  {
    return;
  }

  template <typename Tensor>
  int forward(typename Tensor::data_type alpha,
              Tensor& input,
              Tensor& filter,
              typename Tensor::data_type beta,
              Tensor& output,
              bool skip_halo_exchange = false,
              bool skip_chanfilt_comm = false,
              bool dump_profile = false)
  {
    int_vector paddings, strides;
    const auto& dist = input.get_distribution();
    for (auto i = 0; i < m_num_dims; i++)
    {
      int p = (filter.get_shape()[i] - 1) / 2;
      if (has_halo(dist, i))
        p = 0;
      paddings.push_back(p);
      strides.push_back(0);
    }
    // Note halo exchange not implemented
    for (index_t n = 0; n < input.get_local_shape()[3]; ++n)
    {
      for (index_t k = 0; k < output.get_local_shape()[2]; ++k)
      {
        for (index_t c = 0; c < input.get_local_shape()[2]; ++c)
        {
          ref::apply<Tensor>(alpha,
                             input,
                             n,
                             c,
                             filter,
                             k,
                             c,
                             false,
                             c == 0 ? beta : (typename Tensor::data_type) 1.0,
                             output,
                             n,
                             k,
                             paddings,
                             strides,
                             true);
        }
      }
    }

    return 0;
  }

  template <typename TensorType>
  int apply_bias(typename TensorType::data_type alpha,
                 TensorType& bias,
                 typename TensorType::data_type beta,
                 TensorType& output)
  {
    util::MPIPrintStreamError() << "Not implemented.\n";
    return 1;
  }

  template <typename Tensor>
  int backward_data_exchange_halo(Tensor& d_output)
  {
    util::MPIPrintStreamError() << "Not implemented.\n";
    return 1;
  }

  template <typename Tensor>
  int backward_data(typename Tensor::data_type alpha,
                    Tensor& filter,
                    Tensor& d_output,
                    typename Tensor::data_type beta,
                    Tensor& d_input,
                    bool skip_halo_exchange = false,
                    bool skip_chanfilt_comm = false,
                    bool dump_profile = false)
  {
    // Note halo exchange not implemented

    const auto& dist = d_output.get_distribution();
    int_vector paddings, strides;
    for (auto i = 0; i < m_num_dims; i++)
    {
      int p = (filter.get_shape()[i] - 1) / 2;
      if (has_halo(dist, i))
        p = 0;
      paddings.push_back(p);
      strides.push_back(0);
    }

    for (index_t n = 0; n < d_output.get_local_shape()[3]; ++n)
    {
      for (index_t k = 0; k < d_output.get_local_shape()[2]; ++k)
      {
        for (index_t c = 0; c < filter.get_local_shape()[2]; ++c)
        {
          ref::apply<Tensor>(alpha,
                             d_output,
                             n,
                             k,
                             filter,
                             k,
                             c,
                             true,
                             k == 0 ? beta : (typename Tensor::data_type) 1.0,
                             d_input,
                             n,
                             c,
                             paddings,
                             strides,
                             true);
        }
      }
    }
    return 0;
  }

  template <typename Tensor>
  int backward_filter(typename Tensor::data_type alpha,
                      Tensor& input,
                      Tensor& d_output,
                      typename Tensor::data_type beta,
                      Tensor& d_filter,
                      bool reduce = true,
                      bool skip_chanfilt_comm = false,
                      bool dump_profile = false)
  {
    const auto& dist = input.get_distribution();
    int_vector paddings, strides;
    for (auto i = 0; i < m_num_dims; i++)
    {
      int p = (d_filter.get_shape()[i] - 1) / 2;
      if (has_halo(dist, i))
        p = 0;
      paddings.push_back(p);
      strides.push_back(0);
    }

    for (index_t n = 0; n < input.get_local_shape()[3]; ++n)
    {
      for (index_t k = 0; k < d_output.get_local_shape()[2]; ++k)
      {
        for (index_t c = 0; c < input.get_local_shape()[2]; ++c)
        {
          ref::apply<Tensor>(alpha,
                             input,
                             n,
                             c,
                             d_output,
                             n,
                             k,
                             false,
                             n == 0 ? beta : (typename Tensor::data_type)(1.0),
                             d_filter,
                             k,
                             c,
                             paddings,
                             strides,
                             true);
        }
      }
    }

    if (reduce)
    {
      DISTCONV_CHECK_MPI(
        MPI_Allreduce(MPI_IN_PLACE,
                      d_filter.get_buffer(),
                      d_filter.get_size(),
                      util::get_mpi_data_type<typename Tensor::data_type>(),
                      MPI_SUM,
                      d_filter.get_locale().get_comm()));
    }
    return 0;
  }

  template <typename Tensor>
  int backward_bias(typename Tensor::data_type alpha,
                    Tensor& d_output,
                    typename Tensor::data_type beta,
                    Tensor& bias_gradient,
                    bool reduce = true,
                    bool dump_profile = false)
  {
    util::MPIPrintStreamError() << "Not implemented.\n";
    return 1;
  }

  // Wait for asynchronous tasks
  void wait() {}

  bool is_overlap_fwd_halo_exchange_enabled() const { return false; }

  bool is_overlap_bwd_halo_exchange_enabled() const { return false; }

protected:
  ref::Backend m_be;
  int m_num_dims;
  int_vector m_strides;

  bool has_halo(const tensor::Distribution& dist, int dim)
  {
    return dist.is_distributed(dim) && dist.get_overlap(dim);
  }
};

} // namespace distconv
