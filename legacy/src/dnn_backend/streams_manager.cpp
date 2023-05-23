#include "distconv/dnn_backend/dnn_backend.hpp"
#include "distconv/util/util_gpu.hpp"
#include "distconv_config.hpp"

namespace distconv
{

StreamsManager::StreamsManager(size_t num_streams)
    : StreamsManager(num_streams, h2::gpu::make_stream())
{}

StreamsManager::StreamsManager(size_t num_streams, h2::gpu::DeviceStream stream)
    : m_stream{stream}
{
    m_internal_streams.reserve(num_streams);
    m_priority_streams.reserve(num_streams);
    for (size_t i = 0; i < num_streams; ++i)
        m_internal_streams.push_back(h2::gpu::make_stream_nonblocking());
    for (size_t i = 0; i < num_streams; ++i)
        m_priority_streams.push_back(util::create_priority_stream());
}

StreamsManager::~StreamsManager() noexcept
{
    // FIXME trb: Should track whether to destroy or not.
    // h2::gpu::destroy(m_stream);
    auto const num_streams = m_internal_streams.size();
    for (auto const s : m_priority_streams)
        h2::gpu::destroy(s);
    for (auto const s : m_internal_streams)
        h2::gpu::destroy(s);
}

size_t StreamsManager::num_streams() const noexcept
{
    return m_internal_streams.size();
}

h2::gpu::DeviceStream StreamsManager::stream() const noexcept
{
    return m_stream;
}

h2::gpu::DeviceStream StreamsManager::internal_stream(size_t idx) const
{
    // I don't have a better exception to throw right now...
    return m_internal_streams.at(idx);
}

h2::gpu::DeviceStream StreamsManager::priority_stream(size_t idx) const
{
    return m_priority_streams.at(idx);
}

} // namespace distconv
