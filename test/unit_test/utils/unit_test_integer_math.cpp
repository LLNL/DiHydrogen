////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch.hpp>

#include "h2/utils/IntegerMath.hpp"

#include <type_traits>

using namespace h2;

TEMPLATE_TEST_CASE("ceillog2", "[math][utilities]", uint32_t, uint64_t)
{
    using UInt = TestType;
    static constexpr auto bits = NBits<UInt>;
    static constexpr auto one = static_cast<UInt>(1);

    // 0 is special -- technically log_2(0) does not exist. But if we
    // are looking for "the power of two that bounds the input above",
    // the correct answer is 0.
    CHECK(ceillog2(static_cast<UInt>(0)) == 0);
    CHECK(ceillog2(std::numeric_limits<UInt>::max()) == bits);

    for (auto i = decltype(bits){1}; i < bits; ++i)
    {
        if (i > 1)
            CHECK(ceillog2((one << i) - 1) == i);
        CHECK(ceillog2(one << i) == i);
        CHECK(ceillog2((one << i) + 1) == i + 1);
    }
}

TEMPLATE_TEST_CASE("ispow2", "[math][utilities]", uint32_t, uint64_t)
{
    using UInt = TestType;
    static constexpr auto bits = NBits<UInt>;
    static constexpr auto one = static_cast<UInt>(1);
    for (int i = 0; i < bits; ++i)
    {
        CHECK(ispow2(one << i));
        if (i > 0)
            CHECK_FALSE(ispow2((one << i) + 1));
    }
}

TEST_CASE("host mulhi - uint32_t", "[math][utilities]")
{
    using UInt = uint32_t;
    SECTION("Multiplication fits into 32 bits")
    {
        UInt a = 7, b = 31;
        CHECK(mulhi(a, b) == static_cast<UInt>(0));
    }

    SECTION("Multiplication requires 64 bits")
    {
        UInt x = 123151, y = 6235236;
        CHECK(mulhi(x, y) == 0b10110010U);

        UInt w = 1048575, z = 1048577;
        CHECK(mulhi(w, z) == 0b11111111U);
    }
}

TEST_CASE("host mulhi - uint64_t", "[math][utilities]")
{
    using UInt = uint64_t;
    static constexpr UInt zero = static_cast<UInt>(0);
    SECTION("Multiplication fits into 64 bits")
    {
        // All the uint32_t cases fit in this category now.
        UInt a = 7, b = 31;
        CHECK(mulhi(a, b) == zero);

        UInt x = 123151, y = 6235236;
        CHECK(mulhi(x, y) == zero);

        UInt w = 1048575, z = 1048577;
        CHECK(mulhi(w, z) == zero);
    }

    SECTION("Multiplication requires 128 bits")
    {
        UInt x = 923151289, y = 96235236762;
        CHECK(mulhi(x, y) == 0b100);

        UInt w = 8834007663, z = 42587299100;
        CHECK(mulhi(w, z) == 0b10100);
    }
}

TEMPLATE_TEST_CASE("FastDiv", "[math][utilities]", uint32_t, uint64_t)
{
    using UInt = TestType;
    using FDiv = FastDiv<UInt>;

    UInt const divisor = 17;
    FDiv divmod = FDiv{divisor};
    UInt q, r;
    for (UInt num = 0; num < 100; ++num)
    {
        CHECK_NOTHROW(divmod.divmod(num, q, r));
        CHECK(q == num / divisor);
        CHECK(r == num % divisor);
    }

}
