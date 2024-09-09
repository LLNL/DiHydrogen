#pragma once

#include "distconv/tensor/allreduce.hpp"

#include <Al.hpp>

#include <memory>

namespace distconv
{
namespace tensor
{

template <typename DataType, typename AlBackend>
class AllreduceAl : public Allreduce<DataType>
{
public:
  using AlComm = typename AlBackend::comm_type;
  AllreduceAl(AlComm& comm) : Allreduce<DataType>(), m_comm(comm) {}
  AllreduceAl(std::shared_ptr<AlComm> comm_p)
    : Allreduce<DataType>(), m_comm_p(comm_p), m_comm(*comm_p)
  {}
  virtual ~AllreduceAl() = default;

  virtual void
  allreduce(const DataType* send_buf, DataType* recv_buf, size_t count) override
  {
    Al::Allreduce<AlBackend, DataType>(
      send_buf, recv_buf, count, Al::ReductionOperator::sum, m_comm);
  }
  virtual void allreduce(DataType* buf, size_t count) override
  {
    Al::Allreduce<AlBackend, DataType>(
      buf, count, Al::ReductionOperator::sum, m_comm);
  }

protected:
  std::shared_ptr<AlComm> m_comm_p;
  AlComm& m_comm;
};

template <typename DataType>
using AllreduceAlNCCL = AllreduceAl<DataType, Al::NCCLBackend>;

} // namespace tensor
} // namespace distconv
