#pragma once

#include "distconv/tensor/allreduce.hpp"
#include "distconv/util/util_mpi.hpp"

#include <memory>

namespace distconv
{
namespace tensor
{

template <typename DataType>
class AllreduceMPI : public Allreduce<DataType>
{
public:
  AllreduceMPI(MPI_Comm comm) : Allreduce<DataType>(), m_comm(comm) {}
  virtual ~AllreduceMPI() = default;

  using Allreduce<DataType>::allreduce;

  virtual void
  allreduce(const DataType* send_buf, DataType* recv_buf, size_t count) override
  {
    if (send_buf == recv_buf)
    {
      DISTCONV_CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE,
                                       recv_buf,
                                       count,
                                       util::get_mpi_data_type<DataType>(),
                                       MPI_SUM,
                                       m_comm));
    }
    else
    {
      DISTCONV_CHECK_MPI(MPI_Allreduce(send_buf,
                                       recv_buf,
                                       count,
                                       util::get_mpi_data_type<DataType>(),
                                       MPI_SUM,
                                       m_comm));
    }
  }

protected:
  MPI_Comm m_comm;
};

}  // namespace tensor
}  // namespace distconv
