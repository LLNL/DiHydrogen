#include "h2/core/dispatch.hpp"


// *****
// Static dispatch example (also used in unit testing).

namespace h2
{

namespace impl
{

template <typename T>
void dispatch_test_impl(CPUDev_t, T* v)
{
  *v = 42;
}

#ifdef H2_HAS_GPU
template <typename T>
void dispatch_test_impl(GPUDev_t, T* v) {}
#endif

// Instantiate for all compute types:


template void dispatch_test_impl<float>(CPUDev_t, float*);
template void dispatch_test_impl<double>(CPUDev_t, double*);
template void dispatch_test_impl<std::int32_t>(CPUDev_t, std::int32_t*);
template void dispatch_test_impl<std::uint32_t>(CPUDev_t, std::uint32_t*);

#ifdef H2_HAS_GPU
template void dispatch_test_impl<float>(GPUDev_t, float*);
template void dispatch_test_impl<double>(GPUDev_t, double*);
template void dispatch_test_impl<std::int32_t>(GPUDev_t, std::int32_t*);
template void dispatch_test_impl<std::uint32_t>(GPUDev_t, std::uint32_t*);
#endif

}  // namespace impl

}  // namespace h2

// End static dispatch example.
// *****
