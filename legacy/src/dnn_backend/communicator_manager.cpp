////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "distconv/dnn_backend/dnn_backend.hpp"
#include "distconv/util/util_mpi.hpp" // MPIRootPrintStreamDebug

#define DISTCONV_ASSERT_PTR(ptr)                                               \
  do                                                                           \
  {                                                                            \
    if (!(ptr))                                                                \
    {                                                                          \
      ::distconv::util::PrintStreamError()                                     \
        << "Error at " << __FILE__ << ":" << __LINE__ << ": " << #ptr          \
        << " is a null pointer." << std::endl;                                 \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

namespace distconv
{

CommunicatorManager::CommunicatorManager(MPI_Comm comm,
                                         StreamManager const& stream_mgr)
#ifdef DISTCONV_HAS_P2P
  : m_p2p(comm)
#endif // DISTCONV_HAS_P2P
{
  MPI_Comm_dup(comm, &m_comm);

  auto const num_comms = stream_mgr.num_streams();
  m_internal_comms.reserve(num_comms);
  for (size_t i = 0; i < num_comms; ++i)
  {
    auto const stream = stream_mgr.priority_stream(i);
    m_internal_comms.emplace_back(
      std::make_shared<AlInternalCommType>(m_comm, stream));
  }
  // Impl notes: In an incredibly frustrating twist, it seems that
  // shared_ptr was used here because this class does NOT, in fact,
  // manage the lifetime of these comms. Rather, the references are
  // shared among Convolution objects.
}

MPI_Comm CommunicatorManager::get_comm() const noexcept
{
  return m_comm;
}

auto CommunicatorManager::get_al_nccl_comm() -> AlCommType&
{
  DISTCONV_ASSERT_PTR(m_al_comm);
  return *m_al_comm;
}

size_t CommunicatorManager::get_num_internal_comms() const noexcept
{
  return this->m_internal_comms.size();
}

auto CommunicatorManager::get_internal_comm(size_t idx) const
  -> std::shared_ptr<AlInternalCommType>
{
  return m_internal_comms.at(idx);
}

void CommunicatorManager::init_segmented_ar_comm(size_t seg, MPI_Comm comm)
{
  auto const stream = get_al_nccl_comm().get_stream();
  auto const [_, success] = m_segmented_ar_comms.try_emplace(seg, comm, stream);
  if (success)
    util::MPIPrintStreamDebug()
      << "Set up new segmented AR comm for segments=" << seg;
}
void CommunicatorManager::init_chanfilt_channel_comm(size_t seg, MPI_Comm comm)
{
  auto const stream = get_al_nccl_comm().get_stream();
  auto const [_, success] =
    m_chanfilt_channel_comms.try_emplace(seg, comm, stream);
  if (success)
  {
    auto const& comm = m_chanfilt_channel_comms.at(seg);
    util::MPIPrintStreamDebug()
      << "Setting up new chanfilt channel comm for segments=" << seg
      << " rank=" << comm.rank() << " of " << comm.size();
  }
}

void CommunicatorManager::init_chanfilt_filter_comm(size_t seg, MPI_Comm comm)
{
  auto const stream = get_al_nccl_comm().get_stream();
  auto const [_, success] =
    m_chanfilt_filter_comms.try_emplace(seg, comm, stream);
  if (success)
  {
    auto const& comm = m_chanfilt_filter_comms.at(seg);
    util::MPIPrintStreamDebug()
      << "Setting up new chanfilt filter comm for segments=" << seg
      << " rank=" << comm.rank() << " of " << comm.size();
  }
}

auto CommunicatorManager::get_segmented_ar_comm(size_t idx) -> AlCommType*
{
  if (m_segmented_ar_comms.count(idx) == 0UL)
    return nullptr;
  return &(m_segmented_ar_comms.at(idx));
}

auto CommunicatorManager::get_chanfilt_channel_comm(size_t idx) -> AlCommType*
{
  if (m_chanfilt_channel_comms.count(idx) == 0UL)
    return nullptr;
  return &(m_chanfilt_channel_comms.at(idx));
}

auto CommunicatorManager::get_chanfilt_filter_comm(size_t idx) -> AlCommType*
{
  if (m_chanfilt_filter_comms.count(idx) == 0UL)
    return nullptr;
  return &(m_chanfilt_filter_comms.at(idx));
}

} // namespace distconv
