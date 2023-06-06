#include "distconv/dnn_backend/softmax.hpp"
#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/algorithms_cuda.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/util/util_gpu.hpp"
#include "distconv/util/util_mpi.hpp"

#include <limits>

#if H2_HAS_CUDA
#include <cub/block/block_reduce.cuh>
namespace cubns = cub;
#elif H2_HAS_ROCM
#include <hipcub/block/block_reduce.hpp>
namespace cubns = hipcub;
#endif

using distconv::tensor::CUDAAllocator;
using distconv::tensor::LocaleMPI;

template <typename DataType>
using TensorCUDA = distconv::tensor::Tensor<DataType, LocaleMPI, CUDAAllocator>;

namespace distconv
{
namespace
{

constexpr int block_size = 256;

template <typename DataType>
struct exp;

template <>
struct exp<float>
{
    __device__ __forceinline__ float operator()(float x) const
    {
        return ::expf(x);
    }
};

template <>
struct exp<double>
{
    __device__ __forceinline__ double operator()(double x) const
    {
        return ::exp(x);
    }
};

template <typename DataType>
struct id
{
    __device__ __forceinline__ DataType operator()(DataType x) const
    {
        return x;
    }
};

template <typename DataType>
struct mul
{
    __device__ __forceinline__ DataType operator()(DataType x, DataType y) const
    {
        return x * y;
    }
};

template <typename DataType>
struct div
{
    __device__ __forceinline__ DataType operator()(DataType x, DataType y) const
    {
        return x / y;
    }
};

template <typename DataType>
struct sum
{
    __device__ __forceinline__ DataType init() const { return DataType(0); }
    __device__ __forceinline__ DataType operator()(DataType x, DataType y) const
    {
        return x + y;
    }
};

template <typename DataType>
struct max
{
    __device__ __forceinline__ DataType init() const
    {
        return util::min<DataType>();
    }
    __device__ __forceinline__ DataType operator()(DataType x, DataType y) const
    {
        return (x >= y) ? x : y;
    }
};

template <typename DataType>
struct atomic_max
{
    __device__ __forceinline__ DataType operator()(DataType* addr,
                                                   DataType value) const;
};

template <>
struct atomic_max<float>
{
    __device__ __forceinline__ float operator()(float* addr, float value) const
    {
        float old;
        old =
            (value >= 0)
                ? __int_as_float(atomicMax((int*) addr, __float_as_int(value)))
                : __uint_as_float(
                    atomicMin((unsigned int*) addr, __float_as_uint(value)));
        return old;
    }
};

template <>
struct atomic_max<double>
{
    __device__ __forceinline__ double operator()(double* addr,
                                                 double value) const
    {
        // https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
        long long* addr_as_ll = (long long*) addr;
        long long old = *addr_as_ll, assumed;
        do
        {
            assumed = old;
            auto m = fmax(value, __longlong_as_double(assumed));
            old = atomicCAS((unsigned long long*) addr_as_ll,
                            assumed,
                            (__double_as_longlong(m)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
};

template <typename DataType>
struct atomic_add_fn
{
    __device__ __forceinline__ DataType operator()(DataType* addr,
                                                   DataType value) const
    {
        return atomic_add(addr, value);
    }
};

template <typename DataType>
DataType get_min()
{
    return std::sqrt(std::numeric_limits<DataType>::min());
}

template <typename Tensor>
void set_kernel_params(const Tensor& tensor,
                       int& num_samples,
                       size_t& sample_size,
                       dim3& gdim)
{
    int num_blocks = 80; // == V100 #SMs

    num_samples = tensor.get_local_shape()[-1];
    if (num_samples == 0)
        return;
    sample_size = tensor.get_local_size() / num_samples;
    if (sample_size == 0)
        return;
    int num_blocks_per_sample = util::ceil(num_blocks, num_samples);
    gdim = dim3(num_blocks_per_sample, num_samples);
}

template <typename DataType,
          int BLOCK_SIZE,
          typename Map,
          typename Reduce,
          typename AtomicReduce>
__global__ void reduce_per_sample_kernel(const DataType* __restrict__ x,
                                         size_t sample_size,
                                         Map map,
                                         Reduce reduce,
                                         AtomicReduce atomic_reduce,
                                         DataType* __restrict__ reduction)
{
    auto num_blocks_per_sample = gridDim.x;
    size_t work_per_block =
        (sample_size + sample_size - 1) / num_blocks_per_sample;
    size_t sample_offset = blockIdx.x * work_per_block + threadIdx.x;
    size_t block_end = min((blockIdx.x + 1) * work_per_block, sample_size);
    int sample_idx = blockIdx.y;

    x += sample_idx * sample_size;

    DataType local_sum = reduce.init();
    for (; sample_offset < block_end; sample_offset += BLOCK_SIZE)
    {
        auto x_i = x[sample_offset];
        local_sum = reduce(local_sum, map(x_i));
    }

    using BlockReduce = cubns::BlockReduce<DataType, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    auto block_sum = BlockReduce(temp_storage).Sum(local_sum);
    if (threadIdx.x == 0)
    {
        atomic_reduce(reduction + sample_idx, block_sum);
    }
}

template <typename DataType,
          int BLOCK_SIZE,
          typename Map,
          typename Reduce,
          typename AtomicReduce>
__global__ void reduce_per_sample_kernel(const DataType* __restrict__ x,
                                         const DataType* __restrict__ y,
                                         size_t sample_size,
                                         Map map,
                                         Reduce reduce,
                                         AtomicReduce atomic_reduce,
                                         DataType* __restrict__ reduction)
{
    auto num_blocks_per_sample = gridDim.x;
    size_t work_per_block =
        (sample_size + sample_size - 1) / num_blocks_per_sample;
    size_t sample_offset = blockIdx.x * work_per_block + threadIdx.x;
    size_t block_end = min((blockIdx.x + 1) * work_per_block, sample_size);
    int sample_idx = blockIdx.y;

    x += sample_idx * sample_size;
    y += sample_idx * sample_size;

    DataType local_sum = reduce.init();
    for (; sample_offset < block_end; sample_offset += BLOCK_SIZE)
    {
        auto x_i = x[sample_offset];
        auto y_i = y[sample_offset];
        local_sum = reduce(local_sum, map(x_i, y_i));
    }

    using BlockReduce = cubns::BlockReduce<DataType, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    auto block_sum = BlockReduce(temp_storage).Sum(local_sum);
    if (threadIdx.x == 0)
    {
        atomic_reduce(reduction + sample_idx, block_sum);
    }
}

template <typename DataType, int BLOCK_SIZE, typename Map>
__global__ void
map_per_sample_kernel(const DataType* __restrict__ x,
                      const DataType* __restrict__ y,
                      const DataType* __restrict__ sample_values,
                      size_t sample_size,
                      DataType* __restrict__ z,
                      Map map)
{
    auto num_blocks_per_sample = gridDim.x;
    size_t work_per_block =
        (sample_size + sample_size - 1) / num_blocks_per_sample;
    size_t sample_offset = blockIdx.x * work_per_block + threadIdx.x;
    size_t block_end = min((blockIdx.x + 1) * work_per_block, sample_size);
    int sample_idx = blockIdx.y;

    x += sample_idx * sample_size;
    y += sample_idx * sample_size;
    z += sample_idx * sample_size;

    const auto sample_value = sample_values[sample_idx];

    for (; sample_offset < block_end; sample_offset += BLOCK_SIZE)
    {
        auto x_i = x[sample_offset];
        auto y_i = y[sample_offset];
        auto z_i = map(x_i, y_i, sample_value);
        z[sample_offset] = z_i;
    }
}

template <typename DataType,
          int BLOCK_SIZE,
          typename Map,
          typename Reduce,
          typename AtomicReduce>
__global__ void
map_and_reduce_per_sample_kernel(const DataType* __restrict__ x,
                                 size_t sample_size,
                                 const DataType* __restrict__ sample_values,
                                 Map map,
                                 Reduce reduce,
                                 AtomicReduce atomic_reduce,
                                 DataType* __restrict__ y,
                                 DataType* __restrict__ reduction)
{
    auto num_blocks_per_sample = gridDim.x;
    size_t work_per_block =
        (sample_size + sample_size - 1) / num_blocks_per_sample;
    size_t sample_offset = blockIdx.x * work_per_block + threadIdx.x;
    size_t block_end = min((blockIdx.x + 1) * work_per_block, sample_size);
    int sample_idx = blockIdx.y;

    x += sample_idx * sample_size;
    y += sample_idx * sample_size;

    const auto sample_value = sample_values[sample_idx];

    DataType local_sum = reduce.init();
    for (; sample_offset < block_end; sample_offset += BLOCK_SIZE)
    {
        auto x_i = x[sample_offset];
        x_i = map(x_i, sample_value);
        local_sum = reduce(local_sum, x_i);
        y[sample_offset] = x_i;
    }

    using BlockReduce = cubns::BlockReduce<DataType, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    auto block_sum = BlockReduce(temp_storage).Sum(local_sum);
    if (threadIdx.x == 0)
    {
        atomic_reduce(reduction + sample_idx, block_sum);
    }
}

template <typename DataType, int BLOCK_SIZE, typename Map>
__global__ void
update_per_sample_kernel(DataType* __restrict__ x,
                         const DataType* __restrict__ sample_values,
                         size_t sample_size,
                         Map map)
{
    auto num_blocks_per_sample = gridDim.x;
    size_t work_per_block =
        (sample_size + sample_size - 1) / num_blocks_per_sample;
    size_t sample_offset = blockIdx.x * work_per_block + threadIdx.x;
    size_t block_end = min((blockIdx.x + 1) * work_per_block, sample_size);
    int sample_idx = blockIdx.y;

    x += sample_idx * sample_size;

    const auto sample_val = sample_values[sample_idx];

    for (; sample_offset < block_end; sample_offset += BLOCK_SIZE)
    {
        auto x_i = map(x[sample_offset], sample_val);
        x[sample_offset] = x_i;
    }
}

template <typename Tensor, typename DataType>
void compute_max(const Tensor& tensor,
                 DataType* sample_max,
                 h2::gpu::DeviceStream stream)
{
    dim3 gdim;
    int num_samples;
    size_t sample_size;
    set_kernel_params(tensor, num_samples, sample_size, gdim);

    if (num_samples == 0 || sample_size == 0)
    {
        return;
    }

    reduce_per_sample_kernel<DataType,
                             block_size,
                             id<DataType>,
                             max<DataType>,
                             atomic_max<DataType>>
        <<<gdim, block_size, 0, stream>>>(tensor.get_base_ptr(),
                                          sample_size,
                                          id<DataType>(),
                                          max<DataType>(),
                                          atomic_max<DataType>(),
                                          sample_max);
}

template <typename DataType>
struct exp_shifted
{
    __device__ __forceinline__ DataType operator()(DataType x, DataType y)
    {
        return exp<DataType>()(x - y);
    }
};

template <typename Tensor, typename DataType>
void compute_exp(const Tensor& x,
                 const DataType* sample_max,
                 Tensor& y,
                 DataType* sample_exp,
                 h2::gpu::DeviceStream stream)
{
    dim3 gdim;
    int num_samples;
    size_t sample_size;
    set_kernel_params(x, num_samples, sample_size, gdim);

    if (num_samples == 0 || sample_size == 0)
    {
        return;
    }

    map_and_reduce_per_sample_kernel<DataType,
                                     block_size,
                                     exp_shifted<DataType>,
                                     sum<DataType>,
                                     atomic_add_fn<DataType>>
        <<<gdim, block_size, 0, stream>>>(x.get_base_ptr(),
                                          sample_size,
                                          sample_max,
                                          exp_shifted<DataType>(),
                                          sum<DataType>(),
                                          atomic_add_fn<DataType>(),
                                          y.get_base_ptr(),
                                          sample_exp);
}

template <typename DataType>
struct SoftmaxOp
{
    DataType m_min_output;
    SoftmaxOp(DataType min_output) : m_min_output(min_output) {}
    __device__ __forceinline__ DataType operator()(DataType x, DataType y)
    {
        return ::max(x / y, m_min_output);
    }
};

template <typename Tensor, typename DataType>
void compute_softmax(const DataType* sample_exp,
                     Tensor& output_tensor,
                     h2::gpu::DeviceStream stream)
{
    dim3 gdim;
    int num_samples;
    size_t sample_size;
    set_kernel_params(output_tensor, num_samples, sample_size, gdim);

    if (num_samples == 0 || sample_size == 0)
    {
        return;
    }

    update_per_sample_kernel<DataType, block_size, SoftmaxOp<DataType>>
        <<<gdim, block_size, 0, stream>>>(
            output_tensor.get_base_ptr(),
            sample_exp,
            sample_size,
            SoftmaxOp<DataType>(get_min<DataType>()));
}

template <typename Tensor, typename DataType>
void bp_dotproduct(const Tensor& y,
                   const Tensor& dy,
                   DataType* sample_dp,
                   h2::gpu::DeviceStream stream)
{
    dim3 gdim;
    int num_samples;
    size_t sample_size;
    set_kernel_params(y, num_samples, sample_size, gdim);

    if (num_samples == 0 || sample_size == 0)
    {
        return;
    }

    reduce_per_sample_kernel<DataType,
                             block_size,
                             mul<DataType>,
                             sum<DataType>,
                             atomic_add_fn<DataType>>
        <<<gdim, block_size, 0, stream>>>(y.get_base_ptr(),
                                          dy.get_base_ptr(),
                                          sample_size,
                                          mul<DataType>(),
                                          sum<DataType>(),
                                          atomic_add_fn<DataType>(),
                                          sample_dp);
}

template <typename DataType>
struct bp_compute_func
{
    DataType m_min_output;
    bp_compute_func(DataType min_output) : m_min_output(min_output) {}

    __device__ __forceinline__ DataType operator()(DataType y,
                                                   DataType dy,
                                                   DataType dp)
    {
        if (y > m_min_output)
        {
            return y * (dy - dp);
        }
        else
        {
            return DataType(0);
        }
    }
};

template <typename Tensor, typename DataType>
void bp_compute_gradient(const Tensor& y,
                         const Tensor& dy,
                         DataType* sample_dp,
                         Tensor& dx,
                         h2::gpu::DeviceStream stream)
{
    dim3 gdim;
    int num_samples;
    size_t sample_size;
    set_kernel_params(y, num_samples, sample_size, gdim);

    if (num_samples == 0 || sample_size == 0)
    {
        return;
    }

    map_per_sample_kernel<DataType, block_size, bp_compute_func<DataType>>
        <<<gdim, block_size, 0, stream>>>(
            y.get_base_ptr(),
            dy.get_base_ptr(),
            sample_dp,
            sample_size,
            dx.get_base_ptr(),
            bp_compute_func<DataType>(get_min<DataType>()));
}

template <typename DataType, int BLOCK_SIZE>
__global__ void fp_channel_kernel(const DataType* __restrict__ x,
                                  size_t spatial_size,
                                  int num_channels,
                                  DataType* __restrict__ y)
{
    size_t offset = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t sample_size = spatial_size * num_channels;
    const int sample_idx = blockIdx.y;
    // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
    //
    // Note (trb 02/14/2022): Using __align__(sizeof(DataType)) was
    // causing compiler errors with CUDA 11.4.0 on Pascal. Therefore,
    // I'm hard-coding this to the size of a double, which should be
    // sufficient for all types we use. Also, this:
    // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory/49224531
    extern __shared__ __align__(sizeof(double)) unsigned char x_cache_char[];
    DataType* x_cache = reinterpret_cast<DataType*>(x_cache_char);
    const int cache_idx = threadIdx.x;
    constexpr auto min_output = util::min<DataType>();

    if (offset >= spatial_size)
        return;

    x += sample_idx * sample_size;
    y += sample_idx * sample_size;

    // Calc max
    DataType ch_max = util::min<DataType>();
    for (int cid = 0; cid < num_channels; ++cid)
    {
        auto x_i = x[offset + spatial_size * cid];
        x_cache[cache_idx + BLOCK_SIZE * cid] = x_i;
        ch_max = ::max(ch_max, x_i);
    }

    // Calc exp and sum
    DataType ch_sum = DataType(0);
    for (int cid = 0; cid < num_channels; ++cid)
    {
        auto ch_off = BLOCK_SIZE * cid;
        auto x_i = x_cache[cache_idx + ch_off];
        x_i = exp<DataType>()(x_i - ch_max);
        x_cache[cache_idx + ch_off] = x_i;
        ch_sum += x_i;
    }

    ch_sum = 1 / ch_sum;
    for (int cid = 0; cid < num_channels; ++cid)
    {
        auto ch_off = BLOCK_SIZE * cid;
        auto x_i = x_cache[cache_idx + ch_off];
        x_i = ::max(x_i * ch_sum, min_output);
        y[offset + spatial_size * cid] = x_i;
    }
}

template <typename Tensor>
int fp_channel(const Tensor& x, Tensor& y, h2::gpu::DeviceStream stream)
{
    using DataType = typename Tensor::data_type;

    if (x.get_local_size() == 0)
    {
        return 0;
    }

    auto num_samples = x.get_local_shape()[-1];
    auto num_channels = x.get_local_shape()[-2];
    size_t spatial_size = x.get_local_size() / num_samples / num_channels;
    auto num_blocks_per_sample = util::ceil(spatial_size, (size_t) block_size);

    dim3 gdim(num_blocks_per_sample, num_samples);
    auto shmem_size = num_channels * block_size * sizeof(DataType);

    fp_channel_kernel<DataType, block_size>
        <<<gdim, block_size, shmem_size, stream>>>(
            x.get_base_ptr(), spatial_size, num_channels, y.get_base_ptr());
    DISTCONV_CHECK_GPU(GPU_GET_LAST_ERROR());

    return 0;
}

template <typename DataType, int BLOCK_SIZE>
__global__ void bp_channel_kernel(const DataType* __restrict__ y,
                                  const DataType* __restrict__ dy,
                                  size_t spatial_size,
                                  int num_channels,
                                  DataType* __restrict__ dx)
{
    size_t offset = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t sample_size = spatial_size * num_channels;
    const int sample_idx = blockIdx.y;
    extern __shared__ __align__(sizeof(double)) unsigned char cache_char[];
    DataType* cache = reinterpret_cast<DataType*>(cache_char);
    const int cache_idx = threadIdx.x;
    constexpr auto min_output = util::min<DataType>();

    if (offset >= spatial_size)
        return;

    y += sample_idx * sample_size;
    dy += sample_idx * sample_size;
    dx += sample_idx * sample_size;

    // Calc dotproduct
    DataType dp = DataType(0);
    auto cache_offset = cache_idx;
    for (int cid = 0; cid < num_channels; ++cid)
    {
        auto off = offset + spatial_size * cid;
        auto y_i = y[off];
        auto dy_i = dy[off];
        dp += y_i * dy_i;
        cache[cache_offset] = y_i;
        cache_offset += BLOCK_SIZE;
        cache[cache_offset] = dy_i;
        cache_offset += BLOCK_SIZE;
    }

    // Compute gradients
    cache_offset = cache_idx;
    for (int cid = 0; cid < num_channels; ++cid)
    {
        auto y_i = cache[cache_offset];
        cache_offset += BLOCK_SIZE;
        auto dy_i = cache[cache_offset];
        cache_offset += BLOCK_SIZE;
        auto grad = y_i > min_output ? y_i * (dy_i - dp) : DataType(0);
        dx[offset + spatial_size * cid] = grad;
    }
}

template <typename Tensor>
int bp_channel(const Tensor& y,
               const Tensor& dy,
               Tensor& dx,
               h2::gpu::DeviceStream stream)
{
    using DataType = typename Tensor::data_type;

    if (dx.get_local_size() == 0)
    {
        return 0;
    }

    auto num_samples = dx.get_local_shape()[-1];
    auto num_channels = dx.get_local_shape()[-2];
    size_t spatial_size = dx.get_local_size() / num_samples / num_channels;
    auto num_blocks_per_sample = util::ceil(spatial_size, (size_t) block_size);

    dim3 gdim(num_blocks_per_sample, num_samples);
    auto shmem_size = num_channels * block_size * 2 * sizeof(DataType);

    bp_channel_kernel<DataType, block_size>
        <<<gdim, block_size, shmem_size, stream>>>(y.get_base_ptr(),
                                                   dy.get_base_ptr(),
                                                   spatial_size,
                                                   num_channels,
                                                   dx.get_base_ptr());
    DISTCONV_CHECK_GPU(GPU_GET_LAST_ERROR());

    return 0;
}

} // namespace

template <typename Tensor>
int Softmax<DNNBackend<GPUDNNBackend>>::forward(const Tensor& x, Tensor& y)
{
    using DataType = typename Tensor::data_type;
    util::MPIPrintStreamDebug() << "Softmax FP: " << x << ", " << y;

    auto num_samples = x.get_local_shape()[-1];

    if (num_samples == 0)
    {
        return 0;
    }

    if (m_mode == SoftmaxMode::CHANNEL)
    {
        return fp_channel(x, y, m_stream);
    }

    auto& mempool = internal::RuntimeGPU::get_device_memory_pool();
    auto ws_size = num_samples * sizeof(DataType);
    DataType* sample_max =
        static_cast<DataType*>(mempool.get(ws_size, m_stream));
    DataType* sample_exp =
        static_cast<DataType*>(mempool.get(ws_size, m_stream));

    h2::gpu::mem_zero(sample_max, num_samples, m_stream);
    h2::gpu::mem_zero(sample_exp, num_samples, m_stream);

    // compute sample-wise max
    compute_max(x, sample_max, m_stream);
    allreduce(sample_max, num_samples, true);

    // compute summation of exp
    compute_exp(x, sample_max, y, sample_exp, m_stream);
    allreduce(sample_exp, num_samples, false);

    // update the output
    compute_softmax(sample_exp, y, m_stream);

    mempool.release(sample_max);
    mempool.release(sample_exp);
    return 0;
}

template <typename Tensor>
int Softmax<DNNBackend<GPUDNNBackend>>::backward(const Tensor& y,
                                                 const Tensor& dy,
                                                 Tensor& dx)
{
    using DataType = typename Tensor::data_type;
    util::MPIPrintStreamDebug()
        << "Softmax BP: " << y << ", " << dy << ", " << dx;

    auto num_samples = dx.get_local_shape()[-1];

    if (num_samples == 0)
    {
        return 0;
    }

    if (m_mode == SoftmaxMode::CHANNEL)
    {
        return bp_channel(y, dy, dx, m_stream);
    }

    auto& mempool = internal::RuntimeGPU::get_device_memory_pool();
    auto ws_size = num_samples * sizeof(DataType);

    DataType* sample_dp =
        static_cast<DataType*>(mempool.get(ws_size, m_stream));

    h2::gpu::mem_zero(sample_dp, num_samples, m_stream);

    bp_dotproduct(y, dy, sample_dp, m_stream);
    allreduce(sample_dp, num_samples, false);

    bp_compute_gradient(y, dy, sample_dp, dx, m_stream);

    mempool.release(sample_dp);
    return 0;
}

#define PROTO(T)                                                               \
    template int Softmax<DNNBackend<GPUDNNBackend>>::forward<TensorCUDA<T>>(   \
        const TensorCUDA<T>& x, TensorCUDA<T>& y);                             \
    template int Softmax<DNNBackend<GPUDNNBackend>>::backward<TensorCUDA<T>>(  \
        const TensorCUDA<T>& y, const TensorCUDA<T>& dy, TensorCUDA<T>& dx);
PROTO(float)
PROTO(double)
#undef PROTO

} // namespace distconv
