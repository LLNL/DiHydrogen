#pragma once

#include "distconv_config.hpp"

#include "distconv/base.hpp"
#include "distconv/runtime.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_base.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/vector.hpp"

namespace distconv
{

namespace internal
{
template <typename IntType>
inline IntType get_dilated_filter_size(IntType filter_size, IntType dilation)
{
  return filter_size + (filter_size - IntType(1)) * (dilation - IntType(1));
}

// Assumption: padding must match the size of the filter radius or not
// used at all
template <typename DataType, typename Locale, typename Allocator>
inline void
get_halo_sizes(const tensor::Tensor<DataType, Locale, Allocator>& input,
               const IntVector& filter_dims,
               const IntVector& strides,
               const IntVector& dilations,
               IntVector& fwd_halo_send,
               IntVector& bwd_halo_send,
               IntVector& fwd_halo_recv,
               IntVector& bwd_halo_recv,
               bool with_padding)
{
  const int ND = input.get_num_dims();
  const auto local_shape = input.get_local_shape();
  fwd_halo_send = IntVector(ND, 0);
  bwd_halo_send = IntVector(ND, 0);
  fwd_halo_recv = IntVector(ND, 0);
  bwd_halo_recv = IntVector(ND, 0);
  const auto& split_idx = input.get_split_index();
  const auto& split_shape = input.get_distribution().get_split_shape();
  const auto offset = input.get_global_index();
  // The spatial domains will shrink or expand based on the filter and
  // stride sizes
  for (int i = 0; i < input.get_num_spatial_dims(); ++i)
  {
    // No halo required if not decomposed
    if (split_shape[i] == 1)
      continue;
    auto dilated_filter_dim =
      internal::get_dilated_filter_size(filter_dims[i], dilations[i]);
    // allow even-shaped filter when no halo needed
    if (dilated_filter_dim % 2 == 0)
    {
      assert_eq(strides[i], dilated_filter_dim);
      continue;
    }
    const auto radius = (dilated_filter_dim - 1) / 2;
    const auto s = strides[i];
    const auto off = offset[i];
    // Check the backward direction.
    if (split_idx[i] == 0)
    {
      // beginning of the dimension
      assert0(off);
      // note that when padding is used its size is assumed to be
      // equal to the radius
      if (with_padding)
      {
        bwd_halo_recv[i] = radius;
      }
    }
    else
    {
      assert_always(off > 0);
      // Offset from the first convolution/pooling center. When
      // padding is used, it is the first element, so the offset is
      // simply the offset of the first element of the local
      // tensor. Otherwise, the first center is at offset radius, so
      // the offset from it is the offset of the first local element
      // minus radius.
      const auto x = off - (with_padding ? 0 : radius);
      const auto offset_from_next_stride = (s - (x % s)) % s;
      bwd_halo_recv[i] = radius - offset_from_next_stride;
      const auto xm1 = x - 1;
      const auto offset_from_prev_stride = xm1 % s;
      bwd_halo_send[i] = radius - offset_from_prev_stride;
    }
    // Check the forward direction.
    if (split_idx[i] == split_shape[i] - 1 && !with_padding)
    {
      // end of the dimension. Nothing to send/recv with the forward direction.
      assert_always(off + local_shape[i] == input.get_shape()[i]);
    }
    else
    {
      const auto y = off + local_shape[i] - 1 - (with_padding ? 0 : radius);
      const auto offset_from_prev_stride = y % s;
      fwd_halo_recv[i] = radius - offset_from_prev_stride;
      const auto yp1 = y + 1;
      const auto offset_from_next_stride = (s - (yp1 % s)) % s;
      fwd_halo_send[i] = radius - offset_from_next_stride;
    }

    // Make sure halo sizes are non-negative
    fwd_halo_send[i] = std::max(fwd_halo_send[i], 0);
    fwd_halo_recv[i] = std::max(fwd_halo_recv[i], 0);
    bwd_halo_send[i] = std::max(bwd_halo_send[i], 0);
    bwd_halo_recv[i] = std::max(bwd_halo_recv[i], 0);
  }
}

} // namespace internal

HOST_DEV_FUNC constexpr int get_channel_dim()
{
  return -2;
}

HOST_DEV_FUNC constexpr int get_sample_dim()
{
  return -1;
}

// This is declared here rather than in Distribution as "sample" is a
// notion specific to NN.
inline tensor::Distribution make_sample_distribution(int num_dims,
                                                     int num_procs)
{
  tensor::Shape locale_shape(num_dims, 1);
  locale_shape[get_sample_dim()] = num_procs;
  return tensor::Distribution::make_distribution(locale_shape);
}

inline tensor::Distribution make_strided_sample_distribution(
  int num_dims, const index_t num_samples, int np)
{
  if (num_samples >= (index_t) np)
  {
    return make_sample_distribution(num_dims, np);
  }
  assert0(np % num_samples);
  tensor::Shape proc_shape(num_dims, 1);
  proc_shape[get_sample_dim()] = num_samples;
  proc_shape[0] = np / num_samples;
  auto split_shape = proc_shape;
  split_shape[0] = 1;
  return tensor::Distribution::make_shared_distribution(proc_shape,
                                                        split_shape);
}

template <typename DataType, typename Locale, typename Alloccator>
inline tensor::Shape get_pooling_output_local_tensor_shape(
  const tensor::Tensor<DataType, Locale, Alloccator>& input,
  const int_vector& filter_dims,
  const int_vector& strides,
  bool with_padding,
  const int_vector& dilations)
{
  const int nsd = input.get_num_spatial_dims();
  const auto input_local_shape = input.get_local_shape();
  auto output_local_shape = input.get_local_shape();
  IntVector fwd_halo_send, bwd_halo_send, fwd_halo_recv, bwd_halo_recv;
  internal::get_halo_sizes(input,
                           IntVector(filter_dims),
                           IntVector(strides),
                           IntVector(dilations),
                           fwd_halo_send,
                           bwd_halo_send,
                           fwd_halo_recv,
                           bwd_halo_recv,
                           with_padding);

  for (int i = 0; i < nsd; ++i)
  {
    util::MPIPrintStreamDebug()
      << "i: " << i << ", input_local_shape: " << input_local_shape[i]
      << ", bwd_halo_recv: " << bwd_halo_recv[i]
      << ", fwd_halo_recv: " << fwd_halo_recv[i]
      << ", filter_dims: " << filter_dims[i] << ", padding: " << with_padding;
    int dilated_filter_dim =
      internal::get_dilated_filter_size(filter_dims[i], dilations[i]);
    int dim_with_halo_padding =
      input_local_shape[i] + bwd_halo_recv[i] + fwd_halo_recv[i];
    // Halo size is 0 when not partitioned, but its logical size
    // includes the padding. At this point, padding size is either
    // zero or exact match with the stencil size.
    if (with_padding && input.get_distribution().get_split_shape()[i] == 1)
    {
      dim_with_halo_padding += dilated_filter_dim - 1;
    }
    // Set the local dimension as 0 when the input local shape is too
    // small.
    if (dim_with_halo_padding < dilated_filter_dim)
    {
      output_local_shape[i] = 0;
    }
    else
    {
      output_local_shape[i] =
        util::ceil(dim_with_halo_padding - dilated_filter_dim + 1, strides[i]);
    }
  }
  return output_local_shape;
}

template <typename DataType, typename Locale, typename Allocator>
tensor::Shape get_convolution_output_local_tensor_shape(
  const tensor::Tensor<DataType, Locale, Allocator>& input,
  const int_vector& filter_shape,
  const int_vector& strides,
  bool with_padding,
  const int_vector& dilations,
  int num_groups)
{
  auto output_local_shape = get_pooling_output_local_tensor_shape(
    input, filter_shape, strides, with_padding, dilations);
  // channel size - only if not doing channel parallelism.
  auto input_split_shape = input.get_distribution().get_split_shape();
  if (input_split_shape[-2] == 1)
  {
    assert_eq((int) output_local_shape[-2],
              *(filter_shape.rbegin() + 1) * num_groups);
    output_local_shape[-2] = filter_shape.back();
  }
  else
  {
    assert0(filter_shape.back() % input_split_shape[-2]);
    output_local_shape[-2] = filter_shape.back() / input_split_shape[-2];
  }
  return output_local_shape;
}

template <typename DataType, typename Locale, typename Allocator>
tensor::Shape get_deconvolution_output_local_tensor_shape(
  const tensor::Tensor<DataType, Locale, Allocator>& input,
  const int_vector& filter_dims,
  const int_vector& strides,
  bool with_padding,
  const int_vector& dilations,
  int num_groups)
{
  const int nsd = input.get_num_spatial_dims();
  const auto input_local_shape = input.get_local_shape();
  auto output_local_shape = input.get_local_shape();
  IntVector fwd_halo_send, bwd_halo_send, fwd_halo_recv, bwd_halo_recv;
  internal::get_halo_sizes(input,
                           IntVector(filter_dims),
                           IntVector(strides),
                           IntVector(dilations),
                           fwd_halo_send,
                           bwd_halo_send,
                           fwd_halo_recv,
                           bwd_halo_recv,
                           with_padding);

  for (int i = 0; i < nsd; ++i)
  {
    util::MPIPrintStreamDebug()
      << "i: " << i << ", input_local_shape: " << input_local_shape[i]
      << ", bwd_halo_recv: " << bwd_halo_recv[i]
      << ", fwd_halo_recv: " << fwd_halo_recv[i]
      << ", filter_dims: " << filter_dims[i] << ", padding: " << with_padding;
    int dilated_filter_dim =
      internal::get_dilated_filter_size(filter_dims[i], dilations[i]);
    int dim = (input_local_shape[i] - 1) * strides[i] + dilated_filter_dim;
    dim -= bwd_halo_recv[i] + fwd_halo_recv[i];
    // Halo size is 0 when not partitioned, but its logical size
    // includes the padding. At this point, padding size is either
    // zero or exact match with the stencil size.
    if (with_padding && input.get_distribution().get_split_shape()[i] == 1)
    {
      dim -= dilated_filter_dim - 1;
    }
    output_local_shape[i] = dim;
  }

  // channel size - only if not doing channel parallelism.
  auto input_split_shape = input.get_distribution().get_split_shape();
  if (input_split_shape[-2] == 1)
  {
    assert_eq((int) output_local_shape[-2], filter_dims.back() * num_groups);
    output_local_shape[-2] = *(filter_dims.rbegin() + 1);
  }
  else
  {
    assert0(*(filter_dims.rbegin() + 1) % input_split_shape[-2]);
    output_local_shape[-2] =
      *(filter_dims.rbegin() + 1) / input_split_shape[-2];
  }

  return output_local_shape;
}

template <typename Tensor>
Tensor create_input_tensor(const int_vector& shape,
                           const int_vector& locale_shape,
                           const int_vector& filter_dims,
                           const int_vector& strides,
                           const int_vector& dilations,
                           bool deconv,
                           MPI_Comm comm)
{
  const int nd = shape.size();
  const int nsd = nd - 2;
  IntVector overlap(nd, 0);
  if (!deconv)
  {
    for (int i = 0; i < nsd; ++i)
    {
      if (locale_shape[i] == 1)
        continue;
      auto df = internal::get_dilated_filter_size(filter_dims[i], dilations[i]);
      if (df % 2)
      {
        int overlap_i = (df - 1) / 2;
        overlap[i] = overlap_i;
      }
      else
      {
        // allows even-shaped filters when a stride of the equal size
        // is used
        assert_always(df == strides[i]);
      }
    }
  }
  auto dist = tensor::Distribution::make_overlapped_distribution(
    tensor::Shape(locale_shape), overlap);
  tensor::LocaleMPI loc(comm);
  tensor::Shape division_shape(nd, 0);
  tensor::Shape division_block(nd, 0);
  Tensor t =
    Tensor(tensor::Shape(shape), loc, dist, division_shape, division_block);
  util::MPIPrintStreamDebug() << "Input tensor: " << t;
  return t;
}

template <typename Tensor>
Tensor create_d_input_tensor(const Tensor& input)
{
  Tensor t = Tensor(input.get_shape(),
                    input.get_locale(),
                    input.get_distribution(),
                    input.get_requested_local_shape(),
                    input.get_requested_local_block());
  util::MPIPrintStreamDebug() << "D_input tensor: " << t;
  return t;
}

template <typename Tensor>
Tensor create_filter_tensor(const int_vector& locale_shape,
                            const int_vector& filter_dims,
                            const Tensor& input,
                            index_t num_channels,
                            index_t num_filters,
                            int num_groups,
                            MPI_Comm comm,
                            ChannelParallelismAlgorithm chanfilt_algo,
                            int filter_dim = 0)
{
  const int nd = locale_shape.size();
  const int nsd = nd - 2;
  assert_eq(nsd, (int) filter_dims.size());
  tensor::Shape filter_shape(nd, 0);
  for (int i = 0; i < nsd; ++i)
  {
    filter_shape[i] = filter_dims[i];
  }
  auto filter_locale_shape = tensor::Shape(locale_shape);
  auto split_shape = tensor::Shape(nd, 1);
  if (filter_locale_shape[-2] > 1)
  {
    // Handle channel/filter parallelism.
    assert(num_groups == 1); // No grouped convolution for now.
    if (chanfilt_algo == ChannelParallelismAlgorithm::X)
    {
      filter_locale_shape[-1] = 1;
      split_shape[-2] = filter_locale_shape[-2];
    }
    else if (chanfilt_algo == ChannelParallelismAlgorithm::Y)
    {
      // This is specified with the channel dimension on input.
      filter_locale_shape[-1] = filter_locale_shape[-2];
      filter_locale_shape[-2] = 1;
      split_shape[-1] = filter_locale_shape[-1];
    }
    else if (chanfilt_algo == ChannelParallelismAlgorithm::W)
    {
      if (static_cast<size_t>(filter_dim) > filter_locale_shape[-2]
          || filter_locale_shape[-2] % filter_dim != 0)
      {
        std::cerr << "Invalid filter_dim: channel=" << filter_locale_shape[-2]
                  << " filter=" << filter_dim << "\n";
        abort();
      }
      // The channel dimension of input is split based on
      // filter_dim.
      filter_locale_shape[-1] = filter_dim;
      filter_locale_shape[-2] /= filter_dim;
      split_shape[-1] = filter_locale_shape[-1];
      split_shape[-2] = filter_locale_shape[-2];
    }
  }
  filter_shape[-2] = num_channels / num_groups;
  filter_shape[-1] = num_filters;
  auto dist = tensor::Distribution::make_shared_distribution(
    filter_locale_shape, split_shape);
  util::MPIPrintStreamDebug()
    << "Filter locale shape: " << dist.get_locale_shape()
    << " split shape: " << dist.get_split_shape();
  Tensor t = Tensor(filter_shape, input.get_sub_locale_except_dim(-1), dist);
  util::MPIPrintStreamDebug() << "Filter tensor: " << t;
  return t;
}

template <typename Tensor>
Tensor create_d_filter_tensor(const Tensor& filter)
{
  Tensor t =
    Tensor(filter.get_shape(), filter.get_locale(), filter.get_distribution());
  util::MPIPrintStreamDebug() << "D_filter tensor: " << t;
  return t;
}

template <typename Tensor>
Tensor create_convolution_output_tensor(const Tensor& input,
                                        const Tensor& filter,
                                        const int_vector& strides,
                                        const int_vector& pad,
                                        const int_vector& dilations,
                                        int num_groups)
{
  const int nd = input.get_num_dims();
  const int nsd = input.get_num_spatial_dims();
  const bool use_padding = pad[0] != 0;

  tensor::Shape output_shape(nd, 0);
  for (int i = 0; i < nsd; ++i)
  {
    auto df = internal::get_dilated_filter_size<int>(filter.get_shape()[i],
                                                     dilations[i]);
    assert0((df - 1) % 2);
    assert_always(pad[i] * 2 + 1 == df || pad[i] == 0);
    if (input.get_shape()[i] + pad[i] * 2 < (index_t) df)
    {
      output_shape[i] = 0;
    }
    else
    {
      output_shape[i] = util::ceil(input.get_shape()[i] - df + 1 + pad[i] * 2,
                                   (index_t) strides[i]);
    }
    // padding only for height or width is not considered
    if (use_padding)
    {
      assert_ne((int) pad[i], 0);
    }
    else
    {
      assert0(pad[i]);
    }
  }
  output_shape[-2] = filter.get_shape()[-1];
  output_shape[-1] = input.get_shape()[-1];

  auto dist = input.get_distribution();
  dist.clear_overlap();

  util::MPIPrintStreamDebug()
    << "output_tensor: output_shape: " << output_shape << " dist: " << dist;

  tensor::Shape division_shape = get_convolution_output_local_tensor_shape(
    input,
    filter.get_shape().template get_vector<int>(),
    strides,
    use_padding,
    dilations,
    num_groups);
  tensor::Shape division_block(nd, 0);

  Tensor t = Tensor(
    output_shape, input.get_locale(), dist, division_shape, division_block);
  util::MPIPrintStreamDebug() << "Output tensor: " << t;
  return t;
}

template <typename Tensor>
Tensor create_deconvolution_output_tensor(const Tensor& input,
                                          const Tensor& filter,
                                          const int_vector& strides,
                                          const int_vector& pad,
                                          const int_vector& dilations,
                                          int num_groups)
{
  const int nd = input.get_num_dims();
  const int nsd = input.get_num_spatial_dims();
  const bool use_padding = pad[0] != 0;

  // no padding is assumed
  assert_always(!use_padding);

  tensor::Shape output_shape(nd, 0);
  for (int i = 0; i < nsd; ++i)
  {
    auto df = internal::get_dilated_filter_size<int>(filter.get_shape()[i],
                                                     dilations[i]);
    output_shape[i] = (input.get_shape()[i] - 1) * strides[i] + df;
  }
  output_shape[-2] = filter.get_shape()[-2];
  output_shape[-1] = input.get_shape()[-1];

  auto dist = input.get_distribution();
  dist.clear_overlap();

  util::MPIPrintStreamDebug()
    << "output_tensor: output_shape: " << output_shape << " dist: " << dist;

  tensor::Shape division_shape = get_deconvolution_output_local_tensor_shape(
    input,
    filter.get_shape().template get_vector<int>(),
    strides,
    use_padding,
    dilations,
    num_groups);
  tensor::Shape division_block(nd, 0);

  Tensor t = Tensor(
    output_shape, input.get_locale(), dist, division_shape, division_block);
  util::MPIPrintStreamDebug() << "Output tensor: " << t;
  return t;
}

template <typename Tensor>
Tensor create_convolution_d_output_tensor(const Tensor& output,
                                          const Tensor& filter,
                                          const int_vector& dilations)
{
  const int nd = output.get_num_dims();
  const int nsd = output.get_num_spatial_dims();
  auto dist = output.get_distribution();
  IntVector overlap(nd, 0);

  for (int i = 0; i < nsd; ++i)
  {
    int f = internal::get_dilated_filter_size((int) filter.get_shape()[i],
                                              dilations[i]);
    index_t stencil = (f - 1) / 2;
    assert0((f - 1) % 2);
    if (dist.get_locale_shape()[i] > 1)
    {
      overlap[i] = stencil;
    }
  }
  dist.set_overlap(overlap);
  tensor::Shape division_block(nd, 0);

  Tensor t = Tensor(output.get_shape(),
                    output.get_locale(),
                    dist,
                    output.get_local_shape(),
                    division_block);
  util::MPIPrintStreamDebug() << "D_output tensor: " << t;
  return t;
}

template <typename Tensor>
Tensor create_deconvolution_d_output_tensor(const Tensor& output,
                                            const Tensor& filter,
                                            const int_vector& dilations)
{
  // This only works for the U-Net case
  const int nd = output.get_num_dims();
  auto dist = output.get_distribution();
  IntVector overlap(nd, 0);
  tensor::Shape division_block(nd, 0);
  Tensor t = Tensor(output.get_shape(),
                    output.get_locale(),
                    dist,
                    output.get_local_shape(),
                    division_block);
  util::MPIPrintStreamDebug() << "D_output tensor: " << t;
  return t;
}

template <typename Tensor>
Tensor create_bias_tensor(const Tensor& output)
{
  auto dist = tensor::Distribution::make_shared_distribution(
    output.get_distribution().get_locale_shape());
  tensor::Shape bias_shape(output.get_num_dims(), 1);
  bias_shape[get_channel_dim()] = output.get_shape()[get_channel_dim()];
  Tensor t = Tensor(bias_shape, output.get_locale(), dist);
  util::MPIPrintStreamDebug() << "Bias tensor: " << t;
  return t;
}

template <typename Tensor>
Tensor create_d_bias_tensor(const Tensor& output)
{
  auto t = create_bias_tensor(output);
  util::MPIPrintStreamDebug() << "D_bias tensor: " << t;
  return t;
}

template <typename Tensor>
Tensor create_pooling_output_tensor(const Tensor& input,
                                    const int_vector& window,
                                    const int_vector& strides,
                                    const int_vector& pad)
{
  const int nd = input.get_num_dims();
  const int nsd = input.get_num_spatial_dims();
  bool use_padding = pad[0] != 0;
  auto output_shape = input.get_shape();
  for (int i = 0; i < nsd; ++i)
  {
    if (output_shape[i] + pad[i] * 2 < (index_t) window[i])
    {
      output_shape[i] = 0;
      continue;
    }
    if (window[i] % 2)
    {
      assert_always(pad[i] == 0 || pad[i] * 2 + 1 == window[i]);
      // padding only for height or width is not considered
      if (use_padding)
      {
        assert_ne((int) pad[i], 0);
      }
      else
      {
        assert0(pad[i]);
      }
      output_shape[i] = util::ceil(output_shape[i] - window[i] + 1 + pad[i] * 2,
                                   (index_t) strides[i]);
    }
    else
    {
      assert_always(pad[i] == 0);
      assert_always(strides[i] == window[i]);
      output_shape[i] /= strides[i];
    }
  }
  auto dist = input.get_distribution();
  dist.clear_overlap();

  int_vector dilations(nsd, 1);
  tensor::Shape division_shape = get_pooling_output_local_tensor_shape(
    input, window, strides, use_padding, dilations);
  tensor::Shape division_block(nd, 0);

  util::MPIPrintStreamDebug()
    << "Output tensor. global_shape: " << output_shape
    << ", local shape: " << division_shape << ", dist: " << dist;

  Tensor t = Tensor(
    output_shape, input.get_locale(), dist, division_shape, division_block);
  return t;
}

template <typename Tensor>
Tensor create_pooling_d_output_tensor(const Tensor& output)
{
  Tensor t = Tensor(output.get_shape(),
                    output.get_locale(),
                    output.get_distribution(),
                    output.get_local_shape(),
                    tensor::Shape(output.get_num_dims(), 0));
  util::MPIPrintStreamDebug()
    << "D_output tensor. global_shape: " << t.get_shape()
    << ", local shape: " << t.get_local_shape()
    << ", local real shape: " << t.get_local_real_shape()
    << ", dist: " << t.get_distribution();
  return t;
}

template <typename DataType, typename Alloccator>
inline int dump_tensor(
  const tensor::Tensor<DataType, tensor::LocaleMPI, Alloccator>& t_mpi,
  std::string file_path,
  bool binary = false)
{
  if (binary)
  {
    file_path += ".out";
  }
  else
  {
    file_path += ".txt";
  }

  if (t_mpi.get_locale().get_rank() == 0)
  {
    util::MPIPrintStreamDebug() << "Dumping " << t_mpi << " to " << file_path;
  }

  using TensorProcType =
    tensor::Tensor<DataType, tensor::LocaleProcess, tensor::BaseAllocator>;
  tensor::Distribution proc_dist(t_mpi.get_num_dims());
  TensorProcType temp_tensor(tensor::LocaleProcess(), proc_dist);
  DataType* buf = nullptr;
  bool needs_delete = false;
  if (t_mpi.get_distribution().is_distributed())
  {
    assert0(tensor::Copy(temp_tensor, t_mpi, 0));
    if (t_mpi.get_locale().get_rank() == 0)
    {
      buf = temp_tensor.get_buffer();
    }
  }
  else
  {
    if (t_mpi.get_locale().get_rank() == 0)
    {
      buf = new DataType[t_mpi.get_size()];
      needs_delete = true;
      t_mpi.get_data().copyout(buf);
    }
  }
  if (t_mpi.get_locale().get_rank() == 0)
  {
    std::ofstream out;
    if (!binary)
    {
      out.open(file_path, std::ios::out | std::ios::trunc);
      for (index_t i = 0; i < t_mpi.get_size(); ++i)
      {
        out << buf[i] << std::endl;
      }
    }
    else
    {
      out.open(file_path, std::ios::out | std::ios::trunc | std::ios::binary);
      out.write((char*) buf, t_mpi.get_size() * sizeof(DataType));
    }
    out.close();
  }
  if (needs_delete)
    delete[] buf;
  return 0;
}

template <typename DataType, typename Alloccator>
inline int dump_local_tensor(
  const tensor::Tensor<DataType, tensor::LocaleMPI, Alloccator>& t_mpi,
  std::string file_path,
  bool binary = false)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  file_path += "_" + std::to_string(rank);
  if (binary)
  {
    file_path += ".out";
  }
  else
  {
    file_path += ".txt";
  }

  if (t_mpi.get_locale().get_rank() == 0)
  {
    util::MPIPrintStreamDebug() << "Dumping " << t_mpi << " to " << file_path;
  }

  DataType* buf = new DataType[t_mpi.get_local_size()];
  t_mpi.get_data().copyout(buf);

  std::ofstream out;
  if (!binary)
  {
    out.open(file_path, std::ios::out | std::ios::trunc);
    for (index_t i = 0; i < t_mpi.get_local_size(); ++i)
    {
      out << buf[i] << std::endl;
    }
  }
  else
  {
    out.open(file_path, std::ios::out | std::ios::trunc | std::ios::binary);
    out.write((char*) buf, t_mpi.get_local_size() * sizeof(DataType));
  }
  out.close();
  delete[] buf;

  return 0;
}

} // namespace distconv
