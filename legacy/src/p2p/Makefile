TOP_DIR = ..
include $(TOP_DIR)/config.mk

SRCS = config.cpp request.cpp connection.cpp connection_null.cpp connection_ipc.cpp connection_mpi.cpp mpi.cpp p2p.cpp util_cuda.cpp
OBJS = $(SRCS:.cpp=.o)

CUDA_SRCS = connection_mpi_kernels.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

all: lib

lib: libp2p.a

libp2p.a: $(OBJS) $(CUDA_OBJS)
	$(AR) r $@ $^

%.o:%.cpp
	$(CXX) -c $(CFLAGS) $(MPI_CFLAGS) $(CUDA_CFLAGS) $<

%.o:%.cu
	$(NVCC) -c $(NVCC_CFLAGS) $(MPI_CFLAGS) $<

$(CUDA_SRCS:.cu=.d):%.d:%.cu
	$(NVCC) $(NVCC_CFLAGS) $(MPI_CFLAGS) -M $< > $@
-include $(CUDA_SRCS:.cu=.d)

$(SRCS:.cpp=.d):%.d:%.cpp
	$(CXX) $(CFLAGS) $(MPI_CFLAGS) $(CUDA_CFLAGS) -MM $< > $@
-include $(SRCS:.cpp=.d)

clean:
	$(RM) *.o *.so *.d *.a

.PHONY: install
install: libp2p.a
	mkdir -p $(PREFIX)/lib
	install $< $(PREFIX)/lib

