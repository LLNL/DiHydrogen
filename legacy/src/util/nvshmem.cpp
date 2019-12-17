#include "distconv/util/nvshmem.hpp"
#include "distconv/util/util_mpi.hpp"

namespace distconv {
namespace util {
namespace nvshmem {

#ifdef DISTCONV_HAS_NVSHMEM
void initialize(MPI_Comm comm) {
  util::MPIRootPrintStreamInfo() << "Initializing NVSHMEM with MPI";
  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &comm;
  DISTCONV_CHECK_NVSHMEM(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr));
}

void finalize() {
  util::MPIRootPrintStreamInfo() << "Finalizing NVSHMEM";
  nvshmem_finalize();
}

void barrier() {
  nvshmem_barrier_all();
}
#endif // DISTCONV_HAS_NVSHMEM

} // namespace nvshmem
} // namespace util
} // namespace distconv
