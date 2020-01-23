#include "distconv/base.hpp"
#include "distconv/util/util_cuda.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/tensor/memory_cuda.hpp"
#include "distconv/tensor/allreduce.hpp"
#include "distconv/tensor/allreduce_mpi.hpp"
#include "distconv/tensor/allreduce_mpi_cuda.hpp"
#include "distconv/tensor/allreduce_al.hpp"
#ifdef DISTCONV_HAS_NVSHMEM
#include "distconv/tensor/allreduce_nvshmem.hpp"
#endif // DISTCONV_HAS_NVSHMEM

#include <Al.hpp>
#include <memory>

using DataType = int;
using namespace distconv;

#ifdef DISTCONV_HAS_NVSHMEM
static std::vector<std::string> nvshmem_methods = {
  "AllreduceNVSHMEM",
  "AllreduceNVSHMEMNATIVE",
  "AllreduceNVSHMEMRecursiveDoublingHost",
  "AllreduceNVSHMEMRecursiveDoubling",
  "AllreduceNVSHMEMRecursiveDoublingBuffered",
  "AllreduceNVSHMEMRecursiveDoublingBlock",
};
#endif

bool is_nvshmem_method(const std::string &method) {
#ifdef DISTCONV_HAS_NVSHMEM
  return std::find(nvshmem_methods.begin(), nvshmem_methods.end(), method)
      != nvshmem_methods.end();
#else
  return false;
#endif
}

bool is_nvshmem_method_included(const std::vector<std::string> &methods) {
  for (const auto &m: methods) {
    if (is_nvshmem_method(m)) {
      return true;
    }
  }
  return false;
}

void alloc_buf(const std::string &method, DataType *&ptr, size_t count) {
  if (is_nvshmem_method(method)) {
    util::nvshmem::barrier();
    ptr = static_cast<DataType*>(nvshmem_malloc(sizeof(DataType) * count));
    util::nvshmem::barrier();
    //util::MPIPrintStreamInfo() << "NVSHMEM alloc at: " << ptr;
  } else {
    DISTCONV_CHECK_CUDA(cudaMalloc(&ptr, sizeof(DataType) * count));
  }
}

void free_buf(const std::string &method, void *ptr) {
  if (is_nvshmem_method(method)) {
    //util::MPIPrintStreamInfo() << "Freeing nvshmem buffer: " << ptr;
    util::nvshmem::barrier();
    //util::MPIPrintStreamInfo() << "Freeing nvshmem barrier";
    nvshmem_free(ptr);
    util::nvshmem::barrier();
    //util::MPIPrintStreamInfo() << "Freeing nvshmem done";
  } else {
    DISTCONV_CHECK_CUDA(cudaFree(ptr));
  }
}

std::vector<DataType> create_input(int count, int pid) {
  std::vector<DataType> host(count);
  for (int i = 0; i < count; ++i) {
    host[i] = i + pid;
  }
  return host;
}

void test_setup(const std::string &method,
                int count, DataType *&input_buf, DataType *&output_buf,
                int pid, int np) {
  auto input_host_buf = create_input(count, pid);
  alloc_buf(method, input_buf, count);
  DISTCONV_CHECK_CUDA(cudaMemcpy(
      input_buf, input_host_buf.data(),
      sizeof(DataType) * count, cudaMemcpyHostToDevice));
  alloc_buf(method, output_buf, count);
}

void test_verify(DataType *output_buf, int count, int pid, int np) {
  std::vector<DataType> host(count);
  DISTCONV_CHECK_CUDA(cudaMemcpy(host.data(), output_buf,
                                 sizeof(DataType) * count,
                                 cudaMemcpyDeviceToHost));
  int num_errors = 0;
  int sum_pid = ((np - 1) * np) / 2;
  for (int i = 0; i < count; ++i) {
    auto computed = host[i];
    auto ref = i * np + sum_pid;
    if (ref != computed) {
      ++num_errors;
      // Print error value only for the first entry
      if (num_errors <= 2) {
        util::MPIPrintStreamError() << "Mismatch at " << i << "; ref: " << ref
                                    << ", computed: " << computed;
      }
    }
  }
  int num_max_errors = 0;
  DISTCONV_CHECK_MPI(
      MPI_Allreduce(&num_errors, &num_max_errors, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
  if (num_max_errors > 0) {
    util::MPIRootPrintStreamError() << "Maximum number of mismatches: " << num_max_errors;
    MPI_Finalize();
    std::exit(1);
  }
}

void test_teardown(const std::string &method, DataType *input_buf,
                   DataType *output_buf) {
  free_buf(method, input_buf);
  free_buf(method, output_buf);
  util::MPIPrintStreamInfo() << "test torndown";
}

std::unique_ptr<tensor::Allreduce<DataType>> make_reducer(const std::string name,
                                                          MPI_Comm comm,
                                                          cudaStream_t stream) {
  if (name == "AllreduceMPICUDA") {
    return std::make_unique<tensor::AllreduceMPICUDA<DataType>>(comm, stream);
  } else if (name == "AllreduceAlNCCL") {
    return std::make_unique<tensor::AllreduceAlNCCL<DataType>>(
        std::make_shared<Al::NCCLBackend::comm_type>(comm, stream));
#ifdef DISTCONV_HAS_NVSHMEM
  } else if (name == "AllreduceNVSHMEM") {
    return std::make_unique<tensor::AllreduceNVSHMEM<DataType>>(
        stream, tensor::AllreduceNVSHMEM<DataType>::NAIVE);
  } else if (name == "AllreduceNVSHMEMNATIVE") {
    return std::make_unique<tensor::AllreduceNVSHMEM<DataType>>(
        stream, tensor::AllreduceNVSHMEM<DataType>::NATIVE);
  } else if (name == "AllreduceNVSHMEMRecursiveDoublingHost") {
    return std::make_unique<tensor::AllreduceNVSHMEM<DataType>>(
        stream, tensor::AllreduceNVSHMEM<DataType>::RECURSIVE_DOUBLING_HOST);
  } else if (name == "AllreduceNVSHMEMRecursiveDoubling") {
    return std::make_unique<tensor::AllreduceNVSHMEM<DataType>>(
        stream, tensor::AllreduceNVSHMEM<DataType>::RECURSIVE_DOUBLING);
  } else if (name == "AllreduceNVSHMEMRecursiveDoublingBuffered") {
    return std::make_unique<tensor::AllreduceNVSHMEM<DataType>>(
        stream, tensor::AllreduceNVSHMEM<DataType>::RECURSIVE_DOUBLING_BUFFERED);
  } else if (name == "AllreduceNVSHMEMRecursiveDoublingBlock") {
    return std::make_unique<tensor::AllreduceNVSHMEM<DataType>>(
        stream, tensor::AllreduceNVSHMEM<DataType>::RECURSIVE_DOUBLING_BLOCK);
#endif // DISTCONV_HAS_NVSHMEM
  } else {
    util::MPIRootPrintStreamError() << "Unknown allreducer name: '" << name << "'";
    MPI_Finalize();
    std::exit(1);
  }
}

void test(const std::string &method, int min_count, int max_count, MPI_Comm comm) {
  util::MPIPrintStreamInfo() << "Testing " << method;
  int pid;
  int np;
  MPI_Comm_rank(comm, &pid);
  MPI_Comm_size(comm, &np);
  cudaStream_t stream;
  DISTCONV_CHECK_CUDA(cudaStreamCreate(&stream));
  auto allreducer = make_reducer(method, comm, stream);
  for (int count = min_count; count <= max_count; count *= 2) {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo() << "Count: " << count;
    DataType *input_buf = nullptr;
    DataType *output_buf = nullptr;
    test_setup(method, count, input_buf, output_buf, pid, np);
    assert_always(input_buf != nullptr);
    allreducer->allreduce(input_buf, output_buf, count);
    DISTCONV_CHECK_CUDA(cudaStreamSynchronize(stream));
    test_verify(output_buf, count, pid, np);
    // test inplace
    //allreducer->allreduce(input_buf, count);
    DISTCONV_CHECK_CUDA(cudaStreamSynchronize(stream));
    //test_verify(input_buf, count, pid, np);
    //test_verify(input_buf, 32, pid, np);
    test_teardown(method, input_buf, output_buf);
    util::MPIPrintStreamInfo() << "Count: " << count << " done";
  }
  util::MPIPrintStreamInfo() << "Testing of " << method << " completed";
  MPI_Barrier(MPI_COMM_WORLD);
  DISTCONV_CHECK_CUDA(cudaStreamDestroy(stream));
}

/*
  Usage: mpirun -np N ./test_allreduce_cuda px py [pz], where px
  * py * pz == N
 */
int main(int argc, char *argv[]) {
  int dev = util::choose_gpu();
  cudaSetDevice(dev);
  Al::Initialize(argc, argv);
  int pid;
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  int min_count = 1;
  int max_count = 1024 * 1024;
  std::vector<std::string> methods;
  // default test methods
  methods.push_back("AllreduceMPICUDA");
  methods.push_back("AllreduceAlNCCL");
  int argi = 1;

  if (argi < argc) {
    min_count = atoi(argv[argi]);
    ++argi;
  }
  if (argi < argc) {
    max_count = atoi(argv[argi]);
    ++argi;
  }
  if (argi < argc) {
    methods.clear();
    while (argi < argc) {
      methods.push_back(argv[argi]);
      ++argi;
    }
  }

#ifdef DISTCONV_HAS_NVSHMEM
  // Initialize NVSHMEM when used
  if (is_nvshmem_method_included(methods)) {
    util::nvshmem::initialize(MPI_COMM_WORLD);
  }
#endif

  for (const auto &m: methods) {
    MPI_Barrier(MPI_COMM_WORLD);
    test(m, min_count, max_count, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Completed successfully.";

#ifdef DISTCONV_HAS_NVSHMEM
  // Finalize NVSHMEM when used
  if (is_nvshmem_method_included(methods)) {
    util::nvshmem::finalize();
  }
#endif

  Al::Finalize();
  return 0;
}
