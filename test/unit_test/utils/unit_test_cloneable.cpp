////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <h2/utils/Cloneable.hpp>

#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

namespace
{
using namespace h2;

class Base : public Cloneable<Abstract<Base>>
// class Base : public Cloneable<Base>
{
public:
    Base(int) {}
    virtual ~Base() = default;
    virtual std::string foo() const = 0;
}; // class Base

class Derived1 : public Cloneable<Derived1, Base>
{
public:
    Derived1(int a, float) : DirectBase(a) {}

    std::string foo() const final { return "Derived1"; }
    int derived1_only() { return 1; }

protected:
    using DirectBase = Cloneable<Derived1, Base>;
    using DirectBase::DirectBase;

}; // class Derived1

class Derived2 : public Cloneable<Derived2, Base>
{
public:
    Derived2() : DirectBase(0) {}

    std::string foo() const final { return "Derived2"; }
    int derived2_only() { return 2; }

protected:
    using DirectBase = Cloneable<Derived2, Base>;
    using DirectBase::DirectBase;

}; // class Derived1

} // namespace

TEST_CASE("Testing cloneable with base pointers", "[clone][utilities]")
{
    std::unique_ptr<Base> d1 = std::make_unique<Derived1>(2, 4.f);
    std::unique_ptr<Base> d2 = std::make_unique<Derived2>();

    CHECK(d1->foo() == "Derived1");
    CHECK(d2->foo() == "Derived2");

    auto d1_clone = d1->clone();
    auto d2_clone = d2->clone();

    CHECK(d1_clone->foo() == "Derived1");
    CHECK(d2_clone->foo() == "Derived2");
}

TEST_CASE("Testing cloneable with derived pointers", "[clone][utilities]")
{
    auto d1 = std::make_unique<Derived1>(32, 1.f);
    auto d2 = std::make_unique<Derived2>();

    CHECK(d1->foo() == "Derived1");
    CHECK(d2->foo() == "Derived2");

    auto d1_clone = d1->clone();
    auto d2_clone = d2->clone();

    CHECK(d1_clone->foo() == "Derived1");
    CHECK(d2_clone->foo() == "Derived2");

    CHECK(d1_clone->derived1_only() == 1);
    CHECK(d2_clone->derived2_only() == 2);
}
