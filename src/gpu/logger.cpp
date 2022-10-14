#include "h2/gpu/logger.hpp"

#include <memory>
#include <stdexcept>

#include <spdlog/cfg/env.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#if __has_include(<unistd.h>)
#define HAS_UNISTD_H
#include <unistd.h>
#endif

namespace
{

#ifdef HAS_UNISTD_H
static std::string get_hostname_raw()
{
    char buf[1024];
    if (gethostname(buf, 1024) != 0)
        throw std::runtime_error("gethostname failed.");
    auto end = std::find(buf, buf + 1024, '\0');
    return std::string{buf, end};
}

static std::string const& get_hostname()
{
    static std::string const hostname = get_hostname_raw();
    return hostname;
}
#elif
static std::string const& get_hostname()
{
    static std::string const hostname = "<unknown>";
    return hostname;
}
#endif // HAS_UNISTD_H

class HostnameFlag final : public spdlog::custom_flag_formatter
{
public:
    void format(spdlog::details::log_msg const&,
                std::tm const&,
                spdlog::memory_buf_t& dest) final
    {
        auto const& hostname = get_hostname();
        dest.append(hostname.data(), hostname.data() + hostname.length());
    }

    std::unique_ptr<spdlog::custom_flag_formatter> clone() const final
    {
        return std::make_unique<HostnameFlag>();
    }
}; // class HostnameFlag

// FIXME: This should be setup from a config file.
static std::shared_ptr<spdlog::logger> make_logger()
{
    spdlog::sink_ptr console_sink =
        std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::trace);

    auto formatter = std::make_unique<spdlog::pattern_formatter>();
    formatter->add_flag<HostnameFlag>('h').set_pattern(
        "[%h:%P] [%n:%^%l%$] %v");
    console_sink->set_formatter(std::move(formatter));

    auto logger = std::make_shared<spdlog::logger>(
        std::string{"h2_gpu"}, spdlog::sinks_init_list{console_sink});
    logger->flush_on(spdlog::get_level());
    spdlog::register_logger(logger);
    spdlog::cfg::load_env_levels();

    return logger;
}

} // namespace

spdlog::logger& h2::gpu::logger()
{
    static auto logger = make_logger();
    return *logger;
}
