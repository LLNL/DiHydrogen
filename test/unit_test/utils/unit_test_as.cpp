////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/utils/As.hpp"

#include <cmath>
#include <limits>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("safe_as", "[utilities][as]")
{
    SECTION("Unsigned to signed integers")
    {
        CHECK(h2::safe_as<int32_t>(uint32_t{7}) == 7);
        CHECK_THROWS(
            h2::safe_as<int32_t>(std::numeric_limits<uint32_t>::max()));
    }

    SECTION("Signed to unsigned integers")
    {
        CHECK(h2::safe_as<uint32_t>(1) == uint32_t{1});
        CHECK_THROWS(h2::safe_as<uint32_t>(-1));
    }

    SECTION("Big signed integers")
    {
        CHECK(h2::safe_as<int32_t>(int64_t{1}) == int32_t{1});
        CHECK_THROWS(h2::safe_as<int32_t>(std::numeric_limits<int64_t>::max()));
        CHECK_NOTHROW(
            h2::safe_as<int64_t>(std::numeric_limits<int32_t>::max()));
    }

    SECTION("Big unsigned integers")
    {
        CHECK(h2::safe_as<uint32_t>(uint64_t{1}) == uint32_t{1});
        CHECK_THROWS(
            h2::safe_as<uint32_t>(std::numeric_limits<uint64_t>::max()));
        CHECK_NOTHROW(
            h2::safe_as<uint64_t>(std::numeric_limits<uint32_t>::max()));
    }

    SECTION("Floating point")
    {
        CHECK(h2::safe_as<float>(1.0) == 1.f);
        CHECK_THROWS(h2::safe_as<float>(
            std::nextafter(1.0, std::numeric_limits<double>::max())));
    }

    SECTION("Floating point NaN/inf")
    {
        using fltlim = std::numeric_limits<float>;
        using dbllim = std::numeric_limits<double>;
#ifdef NAN
        CHECK_THROWS(h2::safe_as<double>(NAN));
#endif
        CHECK_THROWS(h2::safe_as<float>(std::nan("")));
        CHECK_THROWS(h2::safe_as<double>(std::nanf("")));
        if constexpr (fltlim::has_quiet_NaN && dbllim::has_quiet_NaN)
        {
            CHECK_THROWS(h2::safe_as<double>(fltlim::quiet_NaN()));
            CHECK_THROWS(h2::safe_as<float>(dbllim::quiet_NaN()));
        }
        if constexpr (fltlim::has_signaling_NaN && dbllim::has_signaling_NaN)
        {
            CHECK_THROWS(h2::safe_as<double>(fltlim::signaling_NaN()));
            CHECK_THROWS(h2::safe_as<float>(dbllim::signaling_NaN()));
        }
        if constexpr (fltlim::has_infinity && dbllim::has_infinity)
        {
            CHECK(h2::safe_as<float>(dbllim::infinity()) == fltlim::infinity());
            CHECK(h2::safe_as<double>(fltlim::infinity()) == dbllim::infinity());
        }
    }
    SECTION("Signed zero")
    {
        CHECK(h2::safe_as<double>(-0.f) == 0.0);
        CHECK(h2::safe_as<float>(-0.) == 0.f);
    }
}
