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
#define PROTO(device, t1) template void dispatch_test_impl<t1>(device, t1*)
H2_INSTANTIATE_1
#undef PROTO

}  // namespace impl

}  // namespace h2

// End static dispatch example.
// *****
