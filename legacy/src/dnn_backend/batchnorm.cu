#include "distconv/distconv.hpp"
#include "distconv/dnn_backend/batchnorm.hpp"
#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/algorithms_cuda.hpp"
#include "distconv/util/util_gpu.hpp"
#include "distconv/util/util_mpi.hpp"

#include <type_traits>

#if H2_HAS_CUDA
#include <cub/block/block_reduce.cuh>
namespace cubns = cub;
#elif H2_HAS_ROCM
#include <hipcub/block/block_reduce.hpp>
namespace cubns = hipcub;
#endif

using distconv::index_t;
using distconv::tensor::CUDAAllocator;
using distconv::tensor::LocaleMPI;
#ifdef DISTCONV_HAS_NVSHMEM
using distconv::tensor::AllreduceNVSHMEM;
using distconv::tensor::AllreduceNVSHMEMDevice;
#endif

using distconv::get_channel_dim;
using distconv::get_sample_dim;

template <typename DataType>
using Tensor = distconv::tensor::Tensor<DataType, LocaleMPI, CUDAAllocator>;

namespace distconv
{
namespace batchnorm
{
namespace
{

template <int ND, typename DataType, int BLOCK_SIZE>
__global__ void
channel_sums_and_sqsums_kernel(const DataType* __restrict__ input,
                               DataType* __restrict__ sums,
                               DataType* __restrict__ sqsums,
                               tensor::Array<ND> shape,
                               tensor::Array<ND> input_strides)
{
    const int tid = threadIdx.x;
    const index_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int ch_idx = blockIdx.y;
    const int num_channels = shape[get_channel_dim()];
    const int num_samples = shape[get_sample_dim()];

    DataType sum = DataType(0);
    DataType sqsum = DataType(0);

    const index_t channel_size = shape.get_size() / num_channels / num_samples;

    if (gidx < channel_size)
    {
        index_t offset = gidx;
        index_t input_offset = 0;
        for (int d = 0; d < ND - 2; ++d)
        {
            int idx = offset % shape[d];
            input_offset += idx * input_strides[d];
            offset /= shape[d];
        }
        input_offset += ch_idx * input_strides[-2];
        for (int s = 0; s < num_samples; ++s)
        {
            const DataType x = input[input_offset];
            sum += x;
            sqsum += x * x;

            input_offset += input_strides[-1];
        }
    }

    using BlockReduce = cubns::BlockReduce<DataType, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage_sum;
    __shared__ typename BlockReduce::TempStorage temp_storage_sqsum;
    sum = BlockReduce(temp_storage_sum).Sum(sum);
    sqsum = BlockReduce(temp_storage_sqsum).Sum(sqsum);
    // Output channel sum to global memory
    if (tid == 0)
    {
        atomic_add(&sums[ch_idx], sum);
        atomic_add(&sqsums[ch_idx], sqsum);
    }
}

template <int ND, typename DataType, int BLOCK_SIZE, typename DataTypeV>
__global__ void
channel_sums_and_sqsums_opt_kernel(const DataTypeV* __restrict__ input,
                                   DataType* __restrict__ sums,
                                   DataType* __restrict__ sqsums,
                                   const int num_channels,
                                   const int num_samples,
                                   const index_t spatial_size,
                                   const index_t spatial_real_size)
{
    const int tid = threadIdx.x;
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int ch_idx = blockIdx.y;
    const auto sample_offset = spatial_real_size * num_channels;

    auto sum = DataType(0);
    auto sqsum = DataType(0);
    index_t offset = spatial_real_size * ch_idx;
    for (int s = 0; s < num_samples; ++s)
    {
        for (int i = idx; i < spatial_size; i += BLOCK_SIZE * gridDim.x)
        {
            const auto x = input[offset + i];
            sum += util::sum(x);
            sqsum += util::sum(x * x);
        }
        offset += sample_offset;
    }

    using BlockReduce = cubns::BlockReduce<DataType, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage_sum;
    __shared__ typename BlockReduce::TempStorage temp_storage_sqsum;
    sum = BlockReduce(temp_storage_sum).Sum(sum);
    sqsum = BlockReduce(temp_storage_sqsum).Sum(sqsum);
    // Output channel sum to global memory
    if (tid == 0)
    {
        atomic_add(&sums[ch_idx], sum);
        atomic_add(&sqsums[ch_idx], sqsum);
    }
}

template <int ND, typename Tensor>
void channel_sums_and_sqsums_opt(int num_samples,
                                 const Tensor& input,
                                 Tensor& sums,
                                 Tensor& sqsums,
                                 h2::gpu::DeviceStream stream)
{
    using DataType = typename Tensor::data_type;

    // Do not contribute to the accumulation if the local tensor is not
    // a split root.
    if (input.get_local_size() == 0 || !input.is_split_root())
        return;

    const int num_channels = input.get_local_shape()[get_channel_dim()];
    constexpr int block_size = 256;
    dim3 block_dim(block_size);
    constexpr index_t thread_work_size = 8;
    constexpr auto block_work_size = block_size * thread_work_size;
    index_t spatial_size = input.get_local_size() / num_channels / num_samples;
    index_t spatial_real_size =
        input.get_local_real_size() / num_channels / num_samples;
    // halo size must be also divisible by a vector width for an
    // alignment requirement
    if (spatial_size % 4 == 0
        && ((spatial_real_size - spatial_size) / 2) % 4 == 0)
    {
        using DataTypeV = typename util::GetVectorType<DataType, 4>::type;
        spatial_size /= 4;
        spatial_real_size /= 4;
        auto num_blocks_per_channel = util::ceil(spatial_size, block_work_size);
        dim3 grid_dim(num_blocks_per_channel, num_channels);
        channel_sums_and_sqsums_opt_kernel<ND, DataType, block_size, DataTypeV>
            <<<grid_dim, block_dim, 0, stream>>>(
                reinterpret_cast<const DataTypeV*>(input.get_const_base_ptr()),
                sums.get_base_ptr(),
                sqsums.get_base_ptr(),
                num_channels,
                num_samples,
                spatial_size,
                spatial_real_size);
    }
    else
    {
        using DataTypeV = DataType;
        auto num_blocks_per_channel = util::ceil(spatial_size, block_work_size);
        dim3 grid_dim(num_blocks_per_channel, num_channels);
        channel_sums_and_sqsums_opt_kernel<ND, DataType, block_size, DataType>
            <<<grid_dim, block_dim, 0, stream>>>(input.get_const_base_ptr(),
                                                 sums.get_base_ptr(),
                                                 sqsums.get_base_ptr(),
                                                 num_channels,
                                                 num_samples,
                                                 spatial_size,
                                                 spatial_real_size);
    }
}

template <int ND, typename Tensor>
void channel_sums_and_sqsums(int num_samples,
                             const Tensor& input,
                             Tensor& sums,
                             Tensor& sqsums,
                             h2::gpu::DeviceStream stream)
{
    using DataType = typename Tensor::data_type;
    // Clear GPU memory
    h2::gpu::mem_zero(sums.get_buffer(), sums.get_local_pitched_size(), stream);
    h2::gpu::mem_zero(
        sqsums.get_buffer(), sqsums.get_local_pitched_size(), stream);

    // Do not contribute to the accumulation if the local tensor is not
    // a split root.
    if (input.get_local_size() == 0 || !input.is_split_root())
        return;

    auto overlap = input.get_overlap();
    bool opt_eligible = true;
    for (int i = 0; i < ND - 3; ++i)
    {
        if (overlap[i] != 0)
        {
            opt_eligible = false;
            break;
        }
    }
    if (std::getenv("DISTCONV_DISABLE_BN_OPT"))
    {
        util::MPIRootPrintStreamInfo() << "Disable BN optimization";
        opt_eligible = false;
    }
    if (opt_eligible)
    {
        channel_sums_and_sqsums_opt<ND, Tensor>(
            num_samples, input, sums, sqsums, stream);
        return;
    }

    const int num_channels = input.get_local_shape()[get_channel_dim()];
    constexpr int block_size = 256;
    dim3 block_dim(block_size);
    index_t channel_size = input.get_local_size() / num_channels / num_samples;
    dim3 grid_dim((channel_size + block_size - 1) / block_size, num_channels);
    auto input_strides = input.get_strides();
    auto shape = input.get_local_shape();
    shape[get_sample_dim()] = num_samples;
    // CUDA grid dimension limitation
    assert_always(num_channels < 65535);

    channel_sums_and_sqsums_kernel<ND, DataType, block_size>
        <<<grid_dim, block_dim, 0, stream>>>(input.get_const_base_ptr(),
                                             sums.get_base_ptr(),
                                             sqsums.get_base_ptr(),
                                             shape,
                                             input_strides);
}
} // namespace

template <typename Tensor>
void channel_sums_and_sqsums(int num_dims,
                             int num_samples,
                             const Tensor& input,
                             Tensor& sums,
                             Tensor& sqsums,
                             h2::gpu::DeviceStream stream)
{
    switch (num_dims)
    {
    case 4:
        channel_sums_and_sqsums<4, Tensor>(
            num_samples, input, sums, sqsums, stream);
        break;
    case 5:
        channel_sums_and_sqsums<5, Tensor>(
            num_samples, input, sums, sqsums, stream);
        break;
    }
}

#define INSTANTIATE_CHANNEL_SUMS_AND_SQSUMS(TYPE)                              \
    template void channel_sums_and_sqsums<Tensor<TYPE>>(                       \
        int num_dims,                                                          \
        int num_samples,                                                       \
        const Tensor<TYPE>& input,                                             \
        Tensor<TYPE>& sums,                                                    \
        Tensor<TYPE>& sqsums,                                                  \
        h2::gpu::DeviceStream stream);
INSTANTIATE_CHANNEL_SUMS_AND_SQSUMS(float)
INSTANTIATE_CHANNEL_SUMS_AND_SQSUMS(double)
#undef INSTANTIATE_CHANNEL_SUMS_AND_SQSUMS

namespace
{

template <typename DataType>
struct sums_to_statistics_functor
{
    index_t m_num_per_sum;
    DataType m_decay;
    sums_to_statistics_functor(index_t num_per_sum, DataType decay)
        : m_num_per_sum(num_per_sum), m_decay(decay)
    {}

    __device__ void operator()(DataType& global_mean,
                               DataType& global_var,
                               DataType& running_mean,
                               DataType& running_var)
    {
        const DataType mean = global_mean / m_num_per_sum;
        const DataType sqmean = global_var / m_num_per_sum;
        DataType var = sqmean - mean * mean;
        var = var > DataType(0) ? var : DataType(0);
        var *= m_num_per_sum / (m_num_per_sum - DataType(1));
        global_mean = mean;
        global_var = var;

        running_mean = m_decay * running_mean + (DataType(1) - m_decay) * mean;
        running_var = m_decay * running_var + (DataType(1) - m_decay) * var;
    }
};

} // namespace

template <typename TensorType>
void sums_to_statistics(index_t num_per_sum,
                        typename TensorType::data_type decay,
                        TensorType& global_mean,
                        TensorType& global_var,
                        TensorType& running_mean,
                        TensorType& running_var,
                        h2::gpu::DeviceStream stream)
{
    using DataType = typename TensorType::data_type;
    if (num_per_sum > 0)
    {
        tensor::Transform(
            global_mean,
            global_var,
            running_mean,
            running_var,
            sums_to_statistics_functor<DataType>(num_per_sum, decay),
            stream);
    }
    else
    {
        // Fill global_var with 1. Do the same thing as the corresponding LBANN
        // code.
        tensor::Transform(
            global_var,
            [] __device__(DataType & global_var) { global_var = DataType(1); },
            stream);
    }
}

#define INSTANTIATE_SUMS_TO_STATISTICS(TYPE)                                   \
    template void sums_to_statistics<Tensor<TYPE>>(                            \
        index_t num_per_sum,                                                   \
        TYPE decay,                                                            \
        Tensor<TYPE> & global_mean,                                            \
        Tensor<TYPE> & global_var,                                             \
        Tensor<TYPE> & running_mean,                                           \
        Tensor<TYPE> & running_var,                                            \
        h2::gpu::DeviceStream stream);
INSTANTIATE_SUMS_TO_STATISTICS(float)
INSTANTIATE_SUMS_TO_STATISTICS(double)
#undef INSTANTIATE_SUMS_TO_STATISTICS

namespace
{

__device__ inline float rsqrt(float x)
{
    return rsqrtf(x);
}

template <int ND, typename DataType>
__global__ void
batch_normalization_kernel(const DataType* __restrict__ input,
                           const DataType* __restrict__ global_mean,
                           const DataType* __restrict__ global_var,
                           const DataType* __restrict__ global_scale,
                           const DataType* __restrict__ global_bias,
                           DataType* __restrict__ output,
                           DataType epsilon,
                           tensor::Array<ND> shape,
                           tensor::Array<ND> input_strides,
                           tensor::Array<ND> output_strides)
{
    const int ch_idx = blockIdx.y;
    const int num_channels = shape[get_channel_dim()];
    const int num_samples = shape[get_sample_dim()];
    const DataType mean = global_mean[ch_idx];
    const DataType var = global_var[ch_idx];
    const DataType scale = global_scale[ch_idx];
    const DataType bias = global_bias[ch_idx];
    const DataType inv_stdev = rsqrt(var + epsilon);

    const index_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
    const index_t channel_size = shape.get_size() / num_channels / num_samples;

    if (gidx < channel_size)
    {
        index_t offset = gidx;
        index_t input_offset = 0, output_offset = 0;
        for (int d = 0; d < ND - 2; ++d)
        {
            int idx = offset % shape[d];
            input_offset += idx * input_strides[d];
            output_offset += idx * output_strides[d];
            offset /= shape[d];
        }
        input_offset += ch_idx * input_strides[-2];
        output_offset += ch_idx * output_strides[-2];
        for (int s = 0; s < num_samples; ++s)
        {
            const DataType x = input[input_offset];
            DataType xhat = (x - mean) * inv_stdev;
            DataType y = scale * xhat + bias;
            output[output_offset] = y;

            input_offset += input_strides[-1];
            output_offset += output_strides[-1];
        }
    }
}

template <int ND, typename DataType, typename DataTypeV>
__global__ void
batch_normalization_opt_kernel(const DataTypeV* __restrict__ input,
                               const DataType* __restrict__ global_mean,
                               const DataType* __restrict__ global_var,
                               const DataType* __restrict__ global_scale,
                               const DataType* __restrict__ global_bias,
                               DataTypeV* __restrict__ output,
                               DataType epsilon,
                               index_t spatial_size,
                               int num_channels)
{
    const auto ch_idx = blockIdx.y;
    const auto sample_idx = blockIdx.z;
    const auto mean = global_mean[ch_idx];
    const auto var = global_var[ch_idx];
    const auto scale = global_scale[ch_idx];
    const auto bias = global_bias[ch_idx];
    const auto inv_stdev = rsqrt(var + epsilon);

    const auto num_threads_per_channel = blockDim.x * gridDim.x;

    auto block_offset = (ch_idx + sample_idx * num_channels) * spatial_size;
    input += block_offset;
    output += block_offset;

    for (index_t idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < spatial_size;
         idx += num_threads_per_channel)
    {
        auto x = input[idx];
        auto xhat = (x - mean) * inv_stdev;
        auto y = xhat * scale + bias;
        output[idx] = y;
    }
}

template <int ND, typename TensorType>
void batch_normalization_opt(int num_samples,
                             const TensorType& input,
                             const TensorType& mean,
                             const TensorType& var,
                             const TensorType& scale,
                             const TensorType& bias,
                             TensorType& output,
                             typename TensorType::data_type epsilon,
                             h2::gpu::DeviceStream stream)
{
    using DataType = typename TensorType::data_type;
    // local tensors can be empty
    if (output.get_local_size() == 0)
        return;
    assert_eq(num_samples, (int) input.get_local_shape()[get_sample_dim()]);
    const int num_channels = input.get_local_shape()[get_channel_dim()];
    constexpr int block_size = 256;
    dim3 block_dim(block_size);
    index_t channel_size = input.get_local_size() / num_channels / num_samples;
    constexpr index_t thread_work_size = 8;
    constexpr auto block_work_size = block_size * thread_work_size;
    if (channel_size % 4 == 0)
    {
        channel_size /= 4;
        auto num_blocks_per_channel = util::ceil(channel_size, block_work_size);
        dim3 grid_dim(num_blocks_per_channel, num_channels, num_samples);
        using DataTypeV = typename util::GetVectorType<DataType, 4>::type;
        batch_normalization_opt_kernel<ND, DataType, DataTypeV>
            <<<grid_dim, block_dim, 0, stream>>>(
                reinterpret_cast<const DataTypeV*>(input.get_const_buffer()),
                mean.get_const_base_ptr(),
                var.get_const_base_ptr(),
                scale.get_const_base_ptr(),
                bias.get_const_base_ptr(),
                reinterpret_cast<DataTypeV*>(output.get_buffer()),
                epsilon,
                channel_size,
                num_channels);
    }
    else
    {
        auto num_blocks_per_channel = util::ceil(channel_size, block_work_size);
        dim3 grid_dim(num_blocks_per_channel, num_channels, num_samples);
        batch_normalization_opt_kernel<ND, DataType, DataType>
            <<<grid_dim, block_dim, 0, stream>>>(input.get_const_buffer(),
                                                 mean.get_const_base_ptr(),
                                                 var.get_const_base_ptr(),
                                                 scale.get_const_base_ptr(),
                                                 bias.get_const_base_ptr(),
                                                 output.get_buffer(),
                                                 epsilon,
                                                 channel_size,
                                                 num_channels);
    }
}

template <int ND, typename TensorType>
void batch_normalization(int num_samples,
                         const TensorType& input,
                         const TensorType& mean,
                         const TensorType& var,
                         const TensorType& scale,
                         const TensorType& bias,
                         TensorType& output,
                         typename TensorType::data_type epsilon,
                         h2::gpu::DeviceStream stream)
{
    using DataType = typename TensorType::data_type;

    if (input.get_local_real_shape() == output.get_local_real_shape())
    {
        if (std::getenv("DISTCONV_DISABLE_BN_OPT"))
        {
            util::MPIRootPrintStreamInfo() << "Disable BN optimization";
        }
        else
        {
            batch_normalization_opt<ND, TensorType>(num_samples,
                                                    input,
                                                    mean,
                                                    var,
                                                    scale,
                                                    bias,
                                                    output,
                                                    epsilon,
                                                    stream);
            return;
        }
    }

    // local tensors can be empty
    if (output.get_local_size() == 0)
        return;
    assert_eq(num_samples, (int) input.get_local_shape()[get_sample_dim()]);
    const int num_channels = input.get_local_shape()[get_channel_dim()];
    constexpr int block_size = 256;
    dim3 block_dim(block_size);
    index_t channel_size = input.get_local_size() / num_channels / num_samples;
    dim3 grid_dim((channel_size + block_size - 1) / block_size, num_channels);
    tensor::Array<ND> input_strides = input.get_strides();
    tensor::Array<ND> output_strides = output.get_strides();
    // CUDA grid dimension limitation
    assert_always(num_channels < 65535);
    tensor::Array<ND> shape = input.get_local_shape();
    batch_normalization_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input.get_const_base_ptr(),
        mean.get_const_base_ptr(),
        var.get_const_base_ptr(),
        scale.get_const_base_ptr(),
        bias.get_const_base_ptr(),
        output.get_base_ptr(),
        epsilon,
        shape,
        input_strides,
        output_strides);
}

} // namespace

template <typename TensorType>
void batch_normalization(int num_dims,
                         int num_samples,
                         const TensorType& input,
                         const TensorType& mean,
                         const TensorType& var,
                         const TensorType& scale,
                         const TensorType& bias,
                         TensorType& output,
                         typename TensorType::data_type epsilon,
                         h2::gpu::DeviceStream stream)
{
    switch (num_dims)
    {
    case 4:
        batch_normalization<4, TensorType>(num_samples,
                                           input,
                                           mean,
                                           var,
                                           scale,
                                           bias,
                                           output,
                                           epsilon,
                                           stream);
        break;
    case 5:
        batch_normalization<5, TensorType>(num_samples,
                                           input,
                                           mean,
                                           var,
                                           scale,
                                           bias,
                                           output,
                                           epsilon,
                                           stream);
        break;
    }
}

#define INSTANTIATE_BATCH_NORMALIZATION(TYPE)                                  \
    template void batch_normalization<Tensor<TYPE>>(                           \
        int num_dims,                                                          \
        int num_samples,                                                       \
        const Tensor<TYPE>& input,                                             \
        const Tensor<TYPE>& mean,                                              \
        const Tensor<TYPE>& var,                                               \
        const Tensor<TYPE>& scale,                                             \
        const Tensor<TYPE>& bias,                                              \
        Tensor<TYPE>& output,                                                  \
        TYPE epsilon,                                                          \
        h2::gpu::DeviceStream stream);
INSTANTIATE_BATCH_NORMALIZATION(float)
INSTANTIATE_BATCH_NORMALIZATION(double)
#undef INSTANTIATE_BATCH_NORMALIZATION

#ifdef DISTCONV_HAS_NVSHMEM
namespace
{

template <int ND,
          typename DataType,
          typename DataType2,
          typename DataTypeV,
          int BLOCK_SIZE>
__global__ void forward_all_kernel(const DataTypeV* __restrict__ input,
                                   DataType* __restrict__ running_mean,
                                   DataType* __restrict__ running_var,
                                   const DataType* __restrict__ scale,
                                   const DataType* __restrict__ bias,
                                   DataTypeV* __restrict__ output,
                                   DataType decay,
                                   DataType epsilon,
                                   const int sample_size,
                                   const int channel_size,
                                   const int spatial_size,
                                   const int spatial_real_size,
                                   const size_t num_per_sum,
                                   AllreduceNVSHMEMDevice<DataType2> ar)
{
    __shared__ DataType2 shared_stat[BLOCK_SIZE];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const auto sample_offset = spatial_real_size * channel_size;

    index_t offset = spatial_real_size * bid;
    DataType2 stat = {DataType(0), DataType(0)};

    for (int s = 0; s < sample_size; ++s)
    {
        for (int i = tid; i < spatial_size; i += BLOCK_SIZE)
        {
            const auto x = input[offset + i];
            stat.x += util::sum(x);
            stat.y += util::sum(x * x);
        }
        offset += sample_offset;
    }

    shared_stat[tid] = stat;

    // Compute channel sum with shared memory reduction
#pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (tid < stride)
        {
            shared_stat[tid] += shared_stat[tid + stride];
        }
    }

    stat = shared_stat[0];

    // Output channel sum to global memory
    const int ch_idx = blockIdx.x;
    if (tid == 0)
    {
        // Assumes only one block per entry
        stat = ar.recursive_doubling_block(stat, 1);
        stat.x = stat.x / num_per_sum;
        stat.y = stat.y / num_per_sum;
        auto v = stat.y - stat.x * stat.x;
        v = max(v, DataType(0));
        v *= num_per_sum / (num_per_sum - DataType(1));
        stat.y = v;
        running_mean[ch_idx] =
            decay * running_mean[ch_idx] + (DataType(1) - decay) * stat.x;
        running_var[ch_idx] =
            decay * running_var[ch_idx] + (DataType(1) - decay) * stat.y;

        stat.y = rsqrt(stat.y + epsilon);
        shared_stat[0] = stat;
    }
    __syncthreads();
    stat = shared_stat[0];

    // fuse the batch_normalization kernel here
    const auto scale_ch = scale[ch_idx];
    const auto bias_ch = bias[ch_idx];

    offset = spatial_real_size * bid;

    for (int s = 0; s < sample_size; ++s)
    {
        for (int i = tid; i < spatial_size; i += BLOCK_SIZE)
        {
            auto idx = offset + i;
            const auto x = input[idx];
            auto xhat = (x - stat.x) * stat.y;
            auto y = xhat * scale_ch + bias_ch;
            output[idx] = y;
        }
        offset += sample_offset;
    }
}

template <int ND, typename Tensor>
void forward_all(const Tensor& input,
                 Tensor& mean,
                 Tensor& var,
                 Tensor& running_mean,
                 Tensor& running_var,
                 Tensor& scale,
                 Tensor& bias,
                 Tensor& output,
                 typename Tensor::data_type decay,
                 typename Tensor::data_type epsilon,
                 h2::gpu::DeviceStream stream,
                 AllreduceNVSHMEM<typename Tensor::data_type>& ar)
{
    using DataType = typename Tensor::data_type;
    using DataType2 = typename util::GetVectorType<DataType, 2>::type;

    const auto shape = input.get_local_shape();
    const auto real_shape = input.get_local_real_shape();
    const int num_samples = shape[get_sample_dim()];
    const int num_channels = shape[get_channel_dim()];

    int spatial_size = shape[0] * shape[1];
    int spatial_real_size = real_shape[0] * real_shape[1];
    if (ND >= 5)
    {
        spatial_size *= shape[2];
        spatial_real_size *= real_shape[2];
    }

    // Assumes halo can only be attached to the outermost spatial
    // dimension
    auto overlap = input.get_overlap();
    assert_eq(overlap[0], 0);
    if (ND >= 5)
    {
        assert_eq(overlap[1], 0);
    }

    constexpr int block_size = 1024;
    dim3 block_dim(block_size);
    dim3 grid_dim(num_channels);
    // CUDA grid dimension limitation
    assert_always(grid_dim.x < 65535);

    ar.recursive_doubling_block_setup(num_channels * 2, 1);

    auto num_per_sum = input.get_size() / input.get_shape()[-2];

    assert_always(input.get_local_size() > 0 && input.is_split_root());

    auto ar_dev = ar.template get_for_device<DataType2>();
    if (spatial_size % 4 == 0 && spatial_real_size % 4 == 0)
    {
        spatial_size /= 4;
        spatial_real_size /= 4;
        using DataTypeV = typename util::GetVectorType<DataType, 4>::type;
        forward_all_kernel<ND, DataType, DataType2, DataTypeV, block_size>
            <<<grid_dim, block_dim, 0, stream>>>(
                reinterpret_cast<const DataTypeV*>(input.get_const_base_ptr()),
                running_mean.get_base_ptr(),
                running_var.get_base_ptr(),
                scale.get_base_ptr(),
                bias.get_base_ptr(),
                reinterpret_cast<DataTypeV*>(output.get_base_ptr()),
                decay,
                epsilon,
                num_samples,
                num_channels,
                spatial_size,
                spatial_real_size,
                num_per_sum,
                ar_dev);
    }
    else
    {
        forward_all_kernel<ND, DataType, DataType2, DataType, block_size>
            <<<grid_dim, block_dim, 0, stream>>>(input.get_const_base_ptr(),
                                                 running_mean.get_base_ptr(),
                                                 running_var.get_base_ptr(),
                                                 scale.get_base_ptr(),
                                                 bias.get_base_ptr(),
                                                 output.get_base_ptr(),
                                                 decay,
                                                 epsilon,
                                                 num_samples,
                                                 num_channels,
                                                 spatial_size,
                                                 spatial_real_size,
                                                 num_per_sum,
                                                 ar_dev);
    }
}

} // namespace

template <typename Tensor>
void forward_all(int num_dims,
                 const Tensor& input,
                 Tensor& mean,
                 Tensor& var,
                 Tensor& running_mean,
                 Tensor& running_var,
                 Tensor& scale,
                 Tensor& bias,
                 Tensor& output,
                 typename Tensor::data_type decay,
                 typename Tensor::data_type epsilon,
                 h2::gpu::DeviceStream stream,
                 AllreduceNVSHMEM<typename Tensor::data_type>& ar)
{
    switch (num_dims)
    {
    case 4:
        forward_all<4, Tensor>(input,
                               mean,
                               var,
                               running_mean,
                               running_var,
                               scale,
                               bias,
                               output,
                               decay,
                               epsilon,
                               stream,
                               ar);
        break;
    case 5:
        forward_all<5, Tensor>(input,
                               mean,
                               var,
                               running_mean,
                               running_var,
                               scale,
                               bias,
                               output,
                               decay,
                               epsilon,
                               stream,
                               ar);
        break;
    }
}

#define INSTANTIATE_FORWARD(TYPE)                                              \
    template void forward_all<Tensor<TYPE>>(int num_dims,                      \
                                            const Tensor<TYPE>& input,         \
                                            Tensor<TYPE>& mean,                \
                                            Tensor<TYPE>& var,                 \
                                            Tensor<TYPE>& running_mean,        \
                                            Tensor<TYPE>& running_var,         \
                                            Tensor<TYPE>& scale,               \
                                            Tensor<TYPE>& bias,                \
                                            Tensor<TYPE>& output,              \
                                            TYPE decay,                        \
                                            TYPE epsilon,                      \
                                            h2::gpu::DeviceStream stream,      \
                                            AllreduceNVSHMEM<TYPE>& ar);
INSTANTIATE_FORWARD(float)
INSTANTIATE_FORWARD(double)
#undef INSTANTIATE_FORWARD
#endif // DISTCONV_HAS_NVSHMEM

namespace
{

template <int ND, typename DataType, int BLOCK_SIZE>
__global__ void backprop1_kernel(const DataType* __restrict__ input,
                                 const DataType* __restrict__ d_output,
                                 const DataType* __restrict__ global_mean,
                                 const DataType* __restrict__ global_var,
                                 const DataType* __restrict__ global_scale,
                                 DataType* __restrict__ global_dscale,
                                 DataType* __restrict__ global_dbias,
                                 DataType* __restrict__ global_dmean,
                                 DataType* __restrict__ global_dvar,
                                 DataType epsilon,
                                 tensor::Array<ND> shape,
                                 tensor::Array<ND> input_strides,
                                 tensor::Array<ND> d_output_strides)
{
    const int tid = threadIdx.x;
    const index_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int ch_idx = blockIdx.y;
    const int num_channels = shape[get_channel_dim()];
    const int num_samples = shape[get_sample_dim()];

    const DataType mean = global_mean[ch_idx];
    const DataType var = global_var[ch_idx];
    const DataType scale = global_scale[ch_idx];
    const DataType inv_stdev = rsqrt(var + epsilon);
    const DataType dvar_factor = inv_stdev * inv_stdev * inv_stdev / 2;

    DataType dscale = DataType(0);
    DataType dbias = DataType(0);
    DataType dmean = DataType(0);
    DataType dvar = DataType(0);

    const index_t channel_size = shape.get_size() / num_channels / num_samples;

    if (gidx < channel_size)
    {
        index_t offset = gidx;
        index_t input_offset = 0, d_output_offset = 0;
        for (int d = 0; d < ND - 2; ++d)
        {
            int idx = offset % shape[d];
            input_offset += idx * input_strides[d];
            d_output_offset += idx * d_output_strides[d];
            offset /= shape[d];
        }
        input_offset += ch_idx * input_strides[-2];
        d_output_offset += ch_idx * d_output_strides[-2];
        for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx)
        {
            const DataType x = input[input_offset];
            const DataType xhat = (x - mean) * inv_stdev;
            const DataType dy = d_output[d_output_offset];
            dscale += dy * xhat;
            dbias += dy;
            const DataType dxhat = dy * scale;
            dmean += -dxhat * inv_stdev;
            dvar += -dxhat * (x - mean) * dvar_factor;

            input_offset += input_strides[-1];
            d_output_offset += d_output_strides[-1];
        }
    }

    using BlockReduce = cubns::BlockReduce<DataType, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage_scale;
    __shared__ typename BlockReduce::TempStorage temp_storage_bias;
    __shared__ typename BlockReduce::TempStorage temp_storage_mean;
    __shared__ typename BlockReduce::TempStorage temp_storage_var;
    dscale = BlockReduce(temp_storage_scale).Sum(dscale);
    dbias = BlockReduce(temp_storage_bias).Sum(dbias);
    dmean = BlockReduce(temp_storage_mean).Sum(dmean);
    dvar = BlockReduce(temp_storage_var).Sum(dvar);

    // Output channel sum to global memory
    if (tid == 0)
    {
        atomic_add(&global_dscale[ch_idx], dscale);
        atomic_add(&global_dbias[ch_idx], dbias);
        atomic_add(&global_dmean[ch_idx], dmean);
        atomic_add(&global_dvar[ch_idx], dvar);
    }
}

template <int ND, typename DataType, int BLOCK_SIZE, typename DataTypeV>
__global__ void backprop1_opt_kernel(const DataTypeV* __restrict__ input,
                                     const DataTypeV* __restrict__ d_output,
                                     const DataType* __restrict__ global_mean,
                                     const DataType* __restrict__ global_var,
                                     const DataType* __restrict__ global_scale,
                                     DataType* __restrict__ global_dscale,
                                     DataType* __restrict__ global_dbias,
                                     DataType* __restrict__ global_dmean,
                                     DataType* __restrict__ global_dvar,
                                     DataType epsilon,
                                     const int num_channels,
                                     const int num_samples,
                                     const index_t spatial_size,
                                     const index_t input_spatial_real_size,
                                     const index_t output_spatial_real_size)
{
    const int tid = threadIdx.x;
    const index_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int ch_idx = blockIdx.y;
    const auto i_sample_offset = input_spatial_real_size * num_channels;
    const auto o_sample_offset = output_spatial_real_size * num_channels;

    const auto mean = global_mean[ch_idx];
    const auto var = global_var[ch_idx];
    const auto scale = global_scale[ch_idx];
    const auto inv_stdev = rsqrt(var + epsilon);
    const auto dvar_factor = inv_stdev * inv_stdev * inv_stdev / 2;

    DataType dscale = DataType(0);
    DataType dbias = DataType(0);
    DataType dmean = DataType(0);
    DataType dvar = DataType(0);

    index_t i_offset = input_spatial_real_size * ch_idx;
    index_t o_offset = output_spatial_real_size * ch_idx;

    for (int s = 0; s < num_samples; ++s)
    {
        for (auto i = idx; i < spatial_size; i += BLOCK_SIZE * gridDim.x)
        {
            const auto x = input[i_offset + i];
            const auto xhat = (x - mean) * inv_stdev;
            const auto dy = d_output[o_offset + i];
            dscale += util::sum(dy * xhat);
            dbias += util::sum(dy);
            const auto dxhat = dy * scale;
            dmean -= util::sum(dxhat * inv_stdev);
            dvar -= util::sum(dxhat * (x - mean) * dvar_factor);
        }
        i_offset += i_sample_offset;
        o_offset += o_sample_offset;
    }

    using BlockReduce = cubns::BlockReduce<DataType, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage_scale;
    __shared__ typename BlockReduce::TempStorage temp_storage_bias;
    __shared__ typename BlockReduce::TempStorage temp_storage_mean;
    __shared__ typename BlockReduce::TempStorage temp_storage_var;
    dscale = BlockReduce(temp_storage_scale).Sum(dscale);
    dbias = BlockReduce(temp_storage_bias).Sum(dbias);
    dmean = BlockReduce(temp_storage_mean).Sum(dmean);
    dvar = BlockReduce(temp_storage_var).Sum(dvar);

    // Output channel sum to global memory
    if (tid == 0)
    {
        atomic_add(&global_dscale[ch_idx], dscale);
        atomic_add(&global_dbias[ch_idx], dbias);
        atomic_add(&global_dmean[ch_idx], dmean);
        atomic_add(&global_dvar[ch_idx], dvar);
    }
}

template <int ND, typename TensorType>
void backprop1_opt(int num_samples,
                   const TensorType& input,
                   const TensorType& d_output,
                   const TensorType& mean,
                   const TensorType& var,
                   const TensorType& scale,
                   TensorType& scale_gradient,
                   TensorType& bias_gradient,
                   TensorType& mean_gradient,
                   TensorType& var_gradient,
                   typename TensorType::data_type epsilon,
                   h2::gpu::DeviceStream stream)
{
    using DataType = typename TensorType::data_type;
    const int num_channels = input.get_local_shape()[get_channel_dim()];
    constexpr int block_size = 256;
    dim3 block_dim(block_size);
    constexpr index_t thread_work_size = 8;
    constexpr auto block_work_size = block_size * thread_work_size;
    index_t spatial_size = input.get_local_size() / num_channels / num_samples;
    index_t i_spatial_real_size =
        input.get_local_real_size() / num_channels / num_samples;
    index_t o_spatial_real_size =
        d_output.get_local_real_size() / num_channels / num_samples;
    // halo size must be also divisible by a vector width for an
    // alignment requirement
    if (spatial_size % 4 == 0
        && ((i_spatial_real_size - spatial_size) / 2) % 4 == 0
        && ((o_spatial_real_size - spatial_size) / 2) % 4 == 0)
    {
        using DataTypeV = typename util::GetVectorType<DataType, 4>::type;
        spatial_size /= 4;
        i_spatial_real_size /= 4;
        o_spatial_real_size /= 4;
        auto num_blocks_per_channel = util::ceil(spatial_size, block_work_size);
        dim3 grid_dim(num_blocks_per_channel, num_channels);
        backprop1_opt_kernel<ND, DataType, block_size, DataTypeV>
            <<<grid_dim, block_dim, 0, stream>>>(
                reinterpret_cast<const DataTypeV*>(input.get_const_base_ptr()),
                reinterpret_cast<const DataTypeV*>(
                    d_output.get_const_base_ptr()),
                mean.get_const_base_ptr(),
                var.get_const_base_ptr(),
                scale.get_const_base_ptr(),
                scale_gradient.get_base_ptr(),
                bias_gradient.get_base_ptr(),
                mean_gradient.get_base_ptr(),
                var_gradient.get_base_ptr(),
                epsilon,
                num_channels,
                num_samples,
                spatial_size,
                i_spatial_real_size,
                o_spatial_real_size);
    }
    else
    {
        using DataTypeV = DataType;
        auto num_blocks_per_channel = util::ceil(spatial_size, block_work_size);
        dim3 grid_dim(num_blocks_per_channel, num_channels);
        backprop1_opt_kernel<ND, DataType, block_size, DataTypeV>
            <<<grid_dim, block_dim, 0, stream>>>(input.get_const_base_ptr(),
                                                 d_output.get_const_base_ptr(),
                                                 mean.get_const_base_ptr(),
                                                 var.get_const_base_ptr(),
                                                 scale.get_const_base_ptr(),
                                                 scale_gradient.get_base_ptr(),
                                                 bias_gradient.get_base_ptr(),
                                                 mean_gradient.get_base_ptr(),
                                                 var_gradient.get_base_ptr(),
                                                 epsilon,
                                                 num_channels,
                                                 num_samples,
                                                 spatial_size,
                                                 i_spatial_real_size,
                                                 o_spatial_real_size);
    }
}

template <int ND, typename TensorType>
void backprop1(int num_samples,
               const TensorType& input,
               const TensorType& d_output,
               const TensorType& mean,
               const TensorType& var,
               const TensorType& scale,
               TensorType& scale_gradient,
               TensorType& bias_gradient,
               TensorType& mean_gradient,
               TensorType& var_gradient,
               typename TensorType::data_type epsilon,
               h2::gpu::DeviceStream stream)
{
    using DataType = typename TensorType::data_type;
    h2::gpu::mem_zero(scale_gradient.get_buffer(),
                      scale_gradient.get_local_pitched_size(),
                      stream);
    h2::gpu::mem_zero(bias_gradient.get_buffer(),
                      bias_gradient.get_local_pitched_size(),
                      stream);
    h2::gpu::mem_zero(mean_gradient.get_buffer(),
                      mean_gradient.get_local_pitched_size(),
                      stream);
    h2::gpu::mem_zero(var_gradient.get_buffer(),
                      var_gradient.get_local_pitched_size(),
                      stream);

    if (input.get_local_size() == 0 || !input.is_split_root())
    {
        return;
    }

    std::vector<IndexVector> overlaps = {input.get_overlap(),
                                         d_output.get_overlap()};
    bool opt_eligible = true;
    for (auto overlap : overlaps)
    {
        for (int i = 0; i < ND - 3; ++i)
        {
            if (overlap[i] != 0)
            {
                opt_eligible = false;
                break;
            }
        }
    }
    if (std::getenv("DISTCONV_DISABLE_BN_OPT"))
    {
        util::MPIRootPrintStreamInfo() << "Disable BN optimization";
        opt_eligible = false;
    }
    if (opt_eligible)
    {
        backprop1_opt<ND, TensorType>(num_samples,
                                      input,
                                      d_output,
                                      mean,
                                      var,
                                      scale,
                                      scale_gradient,
                                      bias_gradient,
                                      mean_gradient,
                                      var_gradient,
                                      epsilon,
                                      stream);
        return;
    }

    const auto input_strides = input.get_strides();
    const auto d_output_strides = d_output.get_strides();
    const int num_channels = input.get_local_shape()[get_channel_dim()];
    // CUDA grid dimension limitation
    assert_always(num_channels < 65535);
    constexpr int block_size = 256;
    dim3 block_dim(block_size);
    auto shape = input.get_local_shape();
    shape[get_sample_dim()] = num_samples;
    index_t channel_size = input.get_local_size() / num_channels / num_samples;
    dim3 grid_dim((channel_size + block_size - 1) / block_size, num_channels);
    backprop1_kernel<ND, DataType, block_size>
        <<<grid_dim, block_dim, 0, stream>>>(input.get_const_base_ptr(),
                                             d_output.get_const_base_ptr(),
                                             mean.get_const_base_ptr(),
                                             var.get_const_base_ptr(),
                                             scale.get_const_base_ptr(),
                                             scale_gradient.get_base_ptr(),
                                             bias_gradient.get_base_ptr(),
                                             mean_gradient.get_base_ptr(),
                                             var_gradient.get_base_ptr(),
                                             epsilon,
                                             shape,
                                             input_strides,
                                             d_output_strides);
}

} // namespace

template <typename TensorType>
void backprop1(int num_dims,
               int num_samples,
               const TensorType& input,
               const TensorType& d_output,
               const TensorType& mean,
               const TensorType& var,
               const TensorType& scale,
               TensorType& scale_gradient,
               TensorType& bias_gradient,
               TensorType& mean_gradient,
               TensorType& var_gradient,
               typename TensorType::data_type epsilon,
               h2::gpu::DeviceStream stream)
{
    switch (num_dims)
    {
    case 4:
        backprop1<4, TensorType>(num_samples,
                                 input,
                                 d_output,
                                 mean,
                                 var,
                                 scale,
                                 scale_gradient,
                                 bias_gradient,
                                 mean_gradient,
                                 var_gradient,
                                 epsilon,
                                 stream);
        break;
    case 5:
        backprop1<5, TensorType>(num_samples,
                                 input,
                                 d_output,
                                 mean,
                                 var,
                                 scale,
                                 scale_gradient,
                                 bias_gradient,
                                 mean_gradient,
                                 var_gradient,
                                 epsilon,
                                 stream);
        break;
    }
}

#define INSTANTIATE_BACKPROP1(TYPE)                                            \
    template void backprop1<Tensor<TYPE>>(int num_dims,                        \
                                          int num_samples,                     \
                                          const Tensor<TYPE>& input,           \
                                          const Tensor<TYPE>& d_output,        \
                                          const Tensor<TYPE>& mean,            \
                                          const Tensor<TYPE>& var,             \
                                          const Tensor<TYPE>& scale,           \
                                          Tensor<TYPE>& scale_gradient,        \
                                          Tensor<TYPE>& bias_gradient,         \
                                          Tensor<TYPE>& mean_gradient,         \
                                          Tensor<TYPE>& var_gradient,          \
                                          TYPE epsilon,                        \
                                          h2::gpu::DeviceStream stream);
INSTANTIATE_BACKPROP1(float)
INSTANTIATE_BACKPROP1(double)
#undef INSTANTIATE_BACKPROP1

namespace
{

template <int ND, typename DataType>
__global__ void backprop2_kernel(const DataType* __restrict__ input,
                                 const DataType* __restrict__ d_output,
                                 const DataType* __restrict__ global_mean,
                                 const DataType* __restrict__ global_var,
                                 const DataType* __restrict__ global_scale,
                                 const DataType* __restrict__ global_dmean,
                                 const DataType* __restrict__ global_dvar,
                                 DataType* __restrict__ d_input,
                                 DataType epsilon,
                                 index_t num_per_sum,
                                 tensor::Array<ND> shape,
                                 tensor::Array<ND> input_strides,
                                 tensor::Array<ND> d_output_strides,
                                 tensor::Array<ND> d_input_strides)
{
    const index_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
    const int ch_idx = blockIdx.y;
    const int num_channels = shape[get_channel_dim()];
    const int num_samples = shape[-1];

    const DataType mean = global_mean[ch_idx];
    const DataType var = global_var[ch_idx];
    const DataType scale = global_scale[ch_idx];
    const DataType dmean = global_dmean[ch_idx];
    const DataType dvar = global_dvar[ch_idx];

    const DataType inv_stdev = rsqrt(var + epsilon);
    const DataType dmean_term = dmean / num_per_sum;
    const DataType dvar_term = dvar * 2 / (num_per_sum - 1);

    const index_t channel_size = shape.get_size() / num_channels / num_samples;

    if (gidx < channel_size)
    {
        index_t offset = gidx;
        index_t input_offset = 0, d_output_offset = 0, d_input_offset = 0;
        for (int d = 0; d < ND - 2; ++d)
        {
            int idx = offset % shape[d];
            input_offset += idx * input_strides[d];
            d_output_offset += idx * d_output_strides[d];
            d_input_offset += idx * d_input_strides[d];
            offset /= shape[d];
        }
        input_offset += ch_idx * input_strides[-2];
        d_output_offset += ch_idx * d_output_strides[-2];
        d_input_offset += ch_idx * d_input_strides[-2];
        for (int s = 0; s < num_samples; ++s)
        {
            const DataType x = input[input_offset];
            const DataType dy = d_output[d_output_offset];
            const DataType dxhat = dy * scale;
            DataType dx = dxhat * inv_stdev;
            dx += dmean_term;
            dx += dvar_term * (x - mean);
            d_input[d_input_offset] = dx;

            input_offset += input_strides[-1];
            d_output_offset += d_output_strides[-1];
            d_input_offset += d_input_strides[-1];
        }
    }
}

template <int ND, typename DataType, typename DataTypeV>
__global__ void backprop2_opt_kernel(const DataTypeV* input,
                                     const DataTypeV* __restrict__ d_output,
                                     const DataType* __restrict__ global_mean,
                                     const DataType* __restrict__ global_var,
                                     const DataType* __restrict__ global_scale,
                                     const DataType* __restrict__ global_dmean,
                                     const DataType* __restrict__ global_dvar,
                                     DataTypeV* d_input,
                                     DataType epsilon,
                                     index_t num_per_sum,
                                     index_t spatial_size,
                                     int num_channels)
{
    const auto ch_idx = blockIdx.y;
    const auto sample_idx = blockIdx.z;
    const auto mean = global_mean[ch_idx];
    const auto var = global_var[ch_idx];
    const auto scale = global_scale[ch_idx];
    const auto dmean = global_dmean[ch_idx];
    const auto dvar = global_dvar[ch_idx];
    const auto inv_stdev = rsqrt(var + epsilon);
    const auto dmean_term = dmean / num_per_sum;
    const auto dvar_term = dvar * 2 / (num_per_sum - 1);

    const auto num_threads_per_channel = blockDim.x * gridDim.x;

    auto block_offset = (ch_idx + sample_idx * num_channels) * spatial_size;
    input += block_offset;
    d_output += block_offset;
    d_input += block_offset;

    for (index_t idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < spatial_size;
         idx += num_threads_per_channel)
    {
        const auto x = input[idx];
        const auto dy = d_output[idx];
        const auto dxhat = dy * scale;
        auto dx = dxhat * inv_stdev;
        dx = dx + dmean_term;
        dx = dx + (x - mean) * dvar_term;
        d_input[idx] = dx;
    }
}

template <int ND, typename TensorType>
void backprop2_opt(index_t num_samples,
                   index_t num_per_sum,
                   const TensorType& input,
                   const TensorType& d_output,
                   const TensorType& mean,
                   const TensorType& var,
                   const TensorType& scale,
                   const TensorType& mean_gradient,
                   const TensorType& var_gradient,
                   TensorType& d_input,
                   typename TensorType::data_type epsilon,
                   h2::gpu::DeviceStream stream)
{
    using DataType = typename TensorType::data_type;
    // local tensors can be empty
    if (input.get_local_size() == 0)
        return;
    assert_eq(num_samples, (int) input.get_local_shape()[get_sample_dim()]);
    const int num_channels = input.get_local_shape()[get_channel_dim()];
    constexpr int block_size = 256;
    dim3 block_dim(block_size);
    index_t channel_size = input.get_local_size() / num_channels / num_samples;
    constexpr index_t thread_work_size = 8;
    constexpr auto block_work_size = block_size * thread_work_size;
    if (channel_size % 4 == 0)
    {
        channel_size /= 4;
        auto num_blocks_per_channel = util::ceil(channel_size, block_work_size);
        dim3 grid_dim(num_blocks_per_channel, num_channels, num_samples);
        using DataTypeV = typename util::GetVectorType<DataType, 4>::type;
        backprop2_opt_kernel<ND, DataType, DataTypeV>
            <<<grid_dim, block_dim, 0, stream>>>(
                reinterpret_cast<const DataTypeV*>(input.get_const_buffer()),
                reinterpret_cast<const DataTypeV*>(d_output.get_const_buffer()),
                mean.get_const_base_ptr(),
                var.get_const_base_ptr(),
                scale.get_const_base_ptr(),
                mean_gradient.get_const_base_ptr(),
                var_gradient.get_const_base_ptr(),
                reinterpret_cast<DataTypeV*>(d_input.get_buffer()),
                epsilon,
                num_per_sum,
                channel_size,
                num_channels);
    }
    else
    {
        auto num_blocks_per_channel = util::ceil(channel_size, block_work_size);
        dim3 grid_dim(num_blocks_per_channel, num_channels, num_samples);
        backprop2_opt_kernel<ND, DataType, DataType>
            <<<grid_dim, block_dim, 0, stream>>>(
                input.get_const_buffer(),
                d_output.get_const_buffer(),
                mean.get_const_base_ptr(),
                var.get_const_base_ptr(),
                scale.get_const_base_ptr(),
                mean_gradient.get_const_base_ptr(),
                var_gradient.get_const_base_ptr(),
                d_input.get_buffer(),
                epsilon,
                num_per_sum,
                channel_size,
                num_channels);
    }
}

template <int ND, typename TensorType>
void backprop2(index_t num_samples,
               index_t num_per_sum,
               const TensorType& input,
               const TensorType& d_output,
               const TensorType& mean,
               const TensorType& var,
               const TensorType& scale,
               const TensorType& mean_gradient,
               const TensorType& var_gradient,
               TensorType& d_input,
               typename TensorType::data_type epsilon,
               h2::gpu::DeviceStream stream)
{
    using DataType = typename TensorType::data_type;

    if (input.get_local_real_shape() == d_output.get_local_real_shape()
        && input.get_local_real_shape() == d_input.get_local_real_shape()
        && input.get_overlap() == 0 && d_output.get_overlap() == 0
        && d_input.get_overlap() == 0)
    {
        if (std::getenv("DISTCONV_DISABLE_BN_OPT"))
        {
            util::MPIRootPrintStreamInfo() << "Disable BN optimization";
        }
        else
        {
            backprop2_opt<ND, TensorType>(num_samples,
                                          num_per_sum,
                                          input,
                                          d_output,
                                          mean,
                                          var,
                                          scale,
                                          mean_gradient,
                                          var_gradient,
                                          d_input,
                                          epsilon,
                                          stream);
            return;
        }
    }

    if (d_input.get_local_size() == 0)
        return;
    const int num_channels = input.get_local_shape()[get_channel_dim()];
    constexpr int block_size = 256;
    dim3 block_dim(block_size);
    index_t channel_size = input.get_local_size() / num_channels / num_samples;
    dim3 grid_dim((channel_size + block_size - 1) / block_size, num_channels);
    auto input_strides = input.get_strides();
    auto d_output_strides = d_output.get_strides();
    auto d_input_strides = d_input.get_strides();
    auto shape = input.get_local_shape();
    shape[get_sample_dim()] = num_samples;
    // CUDA grid dimension limitation
    assert_always(num_channels < 65535);
    backprop2_kernel<ND, DataType>
        <<<grid_dim, block_dim, 0, stream>>>(input.get_const_base_ptr(),
                                             d_output.get_const_base_ptr(),
                                             mean.get_const_base_ptr(),
                                             var.get_const_base_ptr(),
                                             scale.get_const_base_ptr(),
                                             mean_gradient.get_const_base_ptr(),
                                             var_gradient.get_const_base_ptr(),
                                             d_input.get_base_ptr(),
                                             epsilon,
                                             num_per_sum,
                                             shape,
                                             input_strides,
                                             d_output_strides,
                                             d_input_strides);
}

} // namespace

template <typename TensorType>
void backprop2(int num_dims,
               index_t num_samples,
               index_t num_per_sum,
               const TensorType& input,
               const TensorType& d_output,
               const TensorType& mean,
               const TensorType& var,
               const TensorType& scale,
               const TensorType& mean_gradient,
               const TensorType& var_gradient,
               TensorType& d_input,
               typename TensorType::data_type epsilon,
               h2::gpu::DeviceStream stream)
{
    switch (num_dims)
    {
    case 4:
        backprop2<4, TensorType>(num_samples,
                                 num_per_sum,
                                 input,
                                 d_output,
                                 mean,
                                 var,
                                 scale,
                                 mean_gradient,
                                 var_gradient,
                                 d_input,
                                 epsilon,
                                 stream);
        break;
    case 5:
        backprop2<5, TensorType>(num_samples,
                                 num_per_sum,
                                 input,
                                 d_output,
                                 mean,
                                 var,
                                 scale,
                                 mean_gradient,
                                 var_gradient,
                                 d_input,
                                 epsilon,
                                 stream);
        break;
    }
}

#define INSTANTIATE_BACKPROP2(TYPE)                                            \
    template void backprop2<Tensor<TYPE>>(int num_dims,                        \
                                          index_t num_samples,                 \
                                          index_t num_per_sum,                 \
                                          const Tensor<TYPE>& input,           \
                                          const Tensor<TYPE>& d_output,        \
                                          const Tensor<TYPE>& mean,            \
                                          const Tensor<TYPE>& var,             \
                                          const Tensor<TYPE>& scale,           \
                                          const Tensor<TYPE>& mean_gradient,   \
                                          const Tensor<TYPE>& var_gradient,    \
                                          Tensor<TYPE>& d_input,               \
                                          TYPE epsilon,                        \
                                          h2::gpu::DeviceStream stream);
INSTANTIATE_BACKPROP2(float)
INSTANTIATE_BACKPROP2(double)
#undef INSTANTIATE_BACKPROP2

} // namespace batchnorm
} // namespace distconv
