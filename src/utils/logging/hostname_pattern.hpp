////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <unistd.h>

#include "spdlog/pattern_formatter.h"
#include <sys/types.h>

namespace
{
// #ifdef HAS_UNISTD_H
#if 1
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
#else
static std::string const& get_hostname()
{
  static std::string const hostname = "<unknown>";
  return hostname;
}
#endif  // HAS_UNISTD_H

class HostnameFlag final : public spdlog::custom_flag_formatter
{
public:
  void format(spdlog::details::log_msg const&,
              std::tm const&,
              spdlog::memory_buf_t& dest) override
  {
    auto const& hostname = get_hostname();
    dest.append(hostname.data(), hostname.data() + hostname.size());
  }

  std::unique_ptr<custom_flag_formatter> clone() const override
  {
    return spdlog::details::make_unique<HostnameFlag>();
  }
};  // class HostnameFlag

}  // namespace
