////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/utils/environment_vars.hpp"

#include <stdlib.h>
#include <mutex>
#include <unordered_map>

#include "h2/utils/Error.hpp"


namespace
{

struct H2EnvVar
{
  std::mutex mutex;
  std::string value;  /**< Cached value of the variable. */
  std::string default_value;  /**< Default value of the variable. */
  bool is_set;  /**< Whether the variable was set in the environment. */
  bool is_initialized;  /**< Whether we have initialized this entry. */

  H2EnvVar()
  {
    // This is to make unordered_map happy.
    H2_ASSERT_DEBUG(false, "Should not be here");
  }

  H2EnvVar(std::string default_value_)
      : value(""),
        default_value(default_value_),
        is_set(false),
        is_initialized(false)
  {}
};

struct H2EnvManager
{
  H2EnvManager()
  {
    // REGISTER H2 ENV VARS HERE:
    register_h2_env_var("TEST_VAR1", "0", "Used for unit testing");
    register_h2_env_var("TEST_VAR2", "0", "Used for unit testing");
    register_h2_env_var("DEBUG_BACKTRACE",
                        "false",
                        "Whether to always print backtraces in exceptions");
  }

  /**
   * Register an H2 environment variable.
   *
   * @param[in] name The name of the environment variable. It will have
   * "H2_" prepended and be converted to all uppercase.
   * @param[in] default_value The default value of the variable.
   * @param[in] about A textual description of the variable.
   */
  void register_h2_env_var(std::string name,
                           std::string default_value,
                           std::string about);

  /**
   * Ensure an environment variable cache entry is initialized.
   *
   * Assumes the caller holds the cache entry's mutex.
   */
  void ensure_env_var_init(const std::string& name, H2EnvVar& cache_entry);

  bool check_env_var(const std::string& name)
  {
    return env_var_cache.count(name) > 0;
  }

  H2EnvVar& get_env_var(const std::string& name)
  {
    H2_ASSERT_DEBUG(check_env_var(name), "environment variable not registered");

    return env_var_cache[name];
  }

  std::unordered_map<std::string, H2EnvVar> env_var_cache;
};

// Wrapper to use either secure_getenv or getenv.
inline char* raw_getenv(const char* name)
{
#ifdef _GNU_SOURCE
  return secure_getenv(name);
#else
  return getenv(name);
#endif
}

void H2EnvManager::register_h2_env_var(std::string name,
                                       std::string default_value,
                                       [[maybe_unused]] std::string about)
{
  env_var_cache.emplace(name, default_value);
}

void H2EnvManager::ensure_env_var_init(const std::string& name,
                                       H2EnvVar& cache_entry)
{
  if (cache_entry.is_initialized)
  {
    return;
  }

  const std::string h2_name = "H2_" + name;
  const char* env = raw_getenv(h2_name.c_str());
  if (env == nullptr)
  {
    // Variable is not set.
    cache_entry.is_set = false;
    cache_entry.value = cache_entry.default_value;
  }
  else
  {
    // Variable is set.
    cache_entry.is_set = true;
    cache_entry.value = std::string(env);
  }
  cache_entry.is_initialized = true;
}

// Cache for environment variables.
H2EnvManager h2_env_var_cache;

}  // anonymous namespace


namespace h2
{

namespace env
{

bool exists(const std::string& name)
{
  H2EnvVar& cache_entry = h2_env_var_cache.get_env_var(name);
  std::lock_guard<std::mutex> lock(cache_entry.mutex);
  h2_env_var_cache.ensure_env_var_init(name, cache_entry);
  return cache_entry.is_set;
}

std::string get_raw(const std::string& name)
{
  H2EnvVar& cache_entry = h2_env_var_cache.get_env_var(name);
  std::lock_guard<std::mutex> lock(cache_entry.mutex);
  h2_env_var_cache.ensure_env_var_init(name, cache_entry);
  return cache_entry.value;
}

namespace raw
{

bool exists(const std::string& name)
{
  const char* env = raw_getenv(name.c_str());
  return env != nullptr;
}

std::string get_raw(const std::string& name)
{
  const char* env = raw_getenv(name.c_str());
  return std::string((env == nullptr) ? "" : env);
}

}  // namespace raw

}  // namespace env

}  // namespace h2
