////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>

#include "h2/utils/Describable.hpp"

class Foo : public h2::Describable
{
public:
    void short_describe(std::ostream& os) const final
    {
        os << "Foo(short)";
    }
    void describe(std::ostream& os) const final
    {
        os << "Foo(long)";
    }
};

TEST_CASE("Testing describable", "[utilities][describe]")
{
    Foo x;
    SECTION ("Member functions")
    {
        CHECK(x.short_description() == "Foo(short)");
        CHECK(x.description() == "Foo(long)");
    }

    SECTION ("Stream operator")
    {
        std::ostringstream oss;
        oss << x << "|" << describe(x);
        CHECK(oss.str() == x.short_description() + "|" + x.description());
    }
}
