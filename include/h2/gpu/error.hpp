////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 *  This file contains declarations for exception classes for various
 * GPU-related events.
 *
 */

#include "runtime.hpp"

#include <exception>
#include <memory>
#include <sstream>
#include <string>

namespace h2
{
namespace gpu
{

class GPUError : public std::exception
{
public:
    GPUError(DeviceError status) : m_what_str{std::make_shared<std::string>()}
    {
        std::ostringstream msg;
        msg << "GPU error (" << error_name(status)
            << "): " << error_string(status);
        *m_what_str = msg.str();
    }
    GPUError(GPUError const& other) = default;
    GPUError& operator=(GPUError const& other) = default;

    char const* what() const noexcept override { return m_what_str->c_str(); }

private:
    std::shared_ptr<std::string> m_what_str;
}; // class GPUError

class BadGPUAlloc : public GPUError
{
public:
    BadGPUAlloc(DeviceError status, size_t requested_bytes) : GPUError{status}
    {
        (void) requested_bytes;
    }
};

class BadGPUFree : public GPUError
{
public:
    BadGPUFree(DeviceError status) : GPUError{status} {}
};

} // namespace gpu
} // namespace h2
