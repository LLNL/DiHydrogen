#include "distconv/base.hpp"
#include "distconv/util/util_cuda.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/tensor/allreduce.hpp"
#include "distconv/tensor/allreduce_mpi.hpp"
#include "distconv/tensor/allreduce_mpi_cuda.hpp"
#include "distconv/tensor/allreduce_al.hpp"

#include <Al.hpp>
#include <memory>

using DataType = int;
using namespace distconv;

std::vector<DataType> create_input(int count) {
  std::vector<DataType> host(count);
  for (int i = 0; i < count; ++i) {
    host[i] = i;
  }
  return host;
}

void test_setup(int count, DataType *&input_buf, DataType *&output_buf) {
  auto input_host_buf = create_input(count);
  DISTCONV_CHECK_CUDA(cudaMalloc(&input_buf, sizeof(DataType) * count));
  DISTCONV_CHECK_CUDA(cudaMemcpy(input_buf, input_host_buf.data(),
                                 sizeof(DataType) * count, cudaMemcpyHostToDevice));
  DISTCONV_CHECK_CUDA(cudaMalloc(&output_buf, sizeof(DataType) * count));
}

void test_verify(DataType *output_buf, int count, int np) {
  std::vector<DataType> host(count);
  DISTCONV_CHECK_CUDA(cudaMemcpy(host.data(), output_buf,
                                 sizeof(DataType) * count, cudaMemcpyDeviceToHost));
  int num_errors = 0;
  for (int i = 0; i < count; ++i) {
    auto computed = host[i];
    auto ref = i * np;
    if (ref != computed) {
      ++num_errors;
      // Print error value only for the first entry
      if (num_errors == 1) {
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

void test_teardown(DataType *input_buf, DataType * output_buf) {
  DISTCONV_CHECK_CUDA(cudaFree(input_buf));
  DISTCONV_CHECK_CUDA(cudaFree(output_buf));
}

std::unique_ptr<tensor::Allreduce<DataType>> make_reducer(const std::string name,
                                                          MPI_Comm comm,
                                                          cudaStream_t stream) {
  if (name == "AllreduceMPICUDA") {
    return std::make_unique<tensor::AllreduceMPICUDA<DataType>>(comm, stream);
  } else if (name == "AllreduceAlNCCL") {
    return std::make_unique<tensor::AllreduceAlNCCL<DataType>>(
        std::make_shared<Al::NCCLBackend::comm_type>(comm, stream));
  } else {
    util::MPIRootPrintStreamError() << "Unknown allreducer name: '" << name << "'";
    MPI_Finalize();
    std::exit(1);
  }
}

void test(const std::string name, int min_count, int max_count, MPI_Comm comm) {
  int pid;
  int np;
  MPI_Comm_rank(comm, &pid);
  MPI_Comm_size(comm, &np);
  cudaStream_t stream;
  DISTCONV_CHECK_CUDA(cudaStreamCreate(&stream));
  auto allreducer = make_reducer(name, comm, stream);
  for (int count = min_count; count <= max_count; count *= 2) {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo() << "Count: " << count;
    DataType *input_buf = nullptr;
    DataType *output_buf = nullptr;
    test_setup(count, input_buf, output_buf);
    assert_always(input_buf != nullptr);
    allreducer->allreduce(input_buf, output_buf, count);
    test_verify(output_buf, count, np);
    // test inplace
    allreducer->allreduce(input_buf, count);
    test_verify(input_buf, count, np);
    test_teardown(input_buf, output_buf);
  }
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

  for (const auto &m: methods) {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo() << "Testing " << m;
    test(m, min_count, max_count, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Completed successfully.";

  Al::Finalize();
  return 0;
}
