////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_UTILS_ERROR_HPP_
#define H2_UTILS_ERROR_HPP_

#include <h2_config.hpp>

#include <stdexcept>
#include <string>

#include "h2/utils/strings.hpp"

/** @file Error.hpp
 *
 *  A collection of macros and other simple constructs for reporting
 *  and handling errors.
 */

/** @def H2_ADD_FORWARDING_EXCEPTION(name, parent)
 *  @brief Define a class that forwards all arguments to its parent.
 *
 *  This is particularly useful for creating inherited exceptions
 *  from, for example, `std::runtime_error`.
 *
 *  @param name The name of the new class.
 *  @param parent The name of the parent class.
 */
#define H2_DEFINE_FORWARDING_EXCEPTION(name, parent)                         \
  class name : public parent                                                 \
  {                                                                          \
  public:                                                                    \
    /* @brief Constructor */                                                 \
    template <typename... Ts>                                                \
    name(Ts&&... args) : parent(h2::build_string(std::forward<Ts>(args)...)) \
    {}                                                                       \
  }

/** Save a backtrace when constructing an exception. */
static constexpr struct save_backtrace_t {} SaveBacktrace;
/** No not save a backtrace when constructing an exception. */
static constexpr struct no_save_backtrace_t {} NoSaveBacktrace;

/**
 * Base class for H2 exceptions.
 *
 * A stack trace may optionally be recorded. (This is done by default.)
 *
 * @warning This shouldn't need to be a warning, but history shows it
 * needs to be: Do not attempt to use this in a signal handler.
 */
class H2ExceptionBase
{
public:
  H2ExceptionBase(const std::string& what_arg)
    : what_(what_arg)
  {
    if (should_save_backtrace())
    {
      collect_backtrace();
    }
  }

  H2ExceptionBase(const std::string& what_arg, save_backtrace_t)
    : what_(what_arg)
  {
    collect_backtrace();
  }

  H2ExceptionBase(const std::string& what_arg, no_save_backtrace_t)
    : what_(what_arg)
  {}

  virtual ~H2ExceptionBase() {}

  virtual const char* what() const noexcept { return what_.c_str(); }

private:
  std::string what_;  /**< Error message possibly with backtrace. */

  /** Whether to save a backtrace if not explicitly requested. */
  bool should_save_backtrace() const;

  /** Collect backtrace frames and append them to what_. */
  void collect_backtrace();
};

/** Any non-recoverable error. */
class H2FatalException : public H2ExceptionBase
{
public:
  template <typename... Args>
  H2FatalException(Args&&... args)
    : H2ExceptionBase(h2::build_string(std::forward<Args>(args)...),
                      SaveBacktrace)
  {}
};

/**
 * A potentially recoverable error.
 *
 * Collects a backtrace in debug mode or when the H2_DEBUG_BACKTRACE
 * env var is set.
 */
H2_DEFINE_FORWARDING_EXCEPTION(H2NonfatalException, H2ExceptionBase);

/**
 * An alias for H2FatalException.
 */
H2_DEFINE_FORWARDING_EXCEPTION(H2Exception, H2FatalException);

/** @def H2_ASSERT(cond, excptn, msg)
 *  @brief Check that the condition is true and throw an exception if
 *         not.
 *
 *  @param cond The condition to test. Must be a boolean value.
 *  @param excptn The exception to throw if `cond` evaluates to
 *                `false`.
 *  @param ... The arguments to pass to the exception.
 */
#define H2_ASSERT(cond, excptn, ...)                                           \
    do                                                                         \
    {                                                                          \
        if (!(cond))                                                           \
        {                                                                      \
            throw excptn(__VA_ARGS__);                                         \
        }                                                                      \
    } while (0)

#ifdef H2_DEBUG
/** @def H2_ASSERT_DEBUG
 *  @brief Check that a condition is true and throw an exception if
 *         not, but only in a debug build.
 *
 * @param cond Boolean condition to test.
 * @param ... Message to pass to the exception if the condition fails.
 */
#define H2_ASSERT_DEBUG(cond, ...) H2_ASSERT(cond, H2FatalException, __VA_ARGS__)
#else
#define H2_ASSERT_DEBUG(cond, ...)
#endif

/** @def H2_ASSERT_ALWAYS
 * @brief Check that a condition is true and throw an exception if
 *        not regardless of the build type.
 *
 * @param cond Boolean condition to test.
 * @param ... Message to pass to the exception if the condition fails.
 */
#define H2_ASSERT_ALWAYS(cond, ...) H2_ASSERT(cond, H2FatalException, __VA_ARGS__)

namespace h2
{

/** @brief A function to break on when debugging.
 *
 *  @param[in] msg A value that should be available when breaking.
 */
void break_on_me(std::string const& msg = "");

} // namespace h2
#endif // H2_UTILS_ERROR_HPP_
