////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/utils/Cloneable.hpp"

#include <type_traits>

using namespace h2;

class Base : public Cloneable<Abstract<Base>>
// class Base : public Cloneable<Base>
{
public:
    Base(int) {}
    virtual ~Base() = default;
}; // class Base

class Middle : public Cloneable<Abstract<Middle>, Base>
{
protected:
    using DirectBase = Cloneable<Abstract<Middle>, Base>;
    using DirectBase::DirectBase;
};

class Derived1 : public Cloneable<Derived1, Middle>
{
public:
    Derived1(int a, float) : DirectBase(a) {}

    int derived1_only() { return 1; }

protected:
    using DirectBase = Cloneable<Derived1, Middle>;
    using DirectBase::DirectBase;

}; // class Derived1

class Derived2 : public Cloneable<Derived2, Middle>
{
public:
    Derived2() : DirectBase(0) {}

    int derived2_only() { return 2; }

protected:
    using DirectBase = Cloneable<Derived2, Middle>;
    using DirectBase::DirectBase;

}; // class Derived2

template <typename PtrType>
using PtdType = typename PtrType::element_type;

static_assert(
    std::is_same_v<PtdType<decltype(std::declval<Base>().clone())>, Base>, "");
static_assert(
    std::is_same_v<PtdType<decltype(std::declval<Middle>().clone())>, Middle>,
    "");
static_assert(
    std::is_same_v<PtdType<decltype(std::declval<Derived1>().clone())>,
                   Derived1>,
    "");
static_assert(
    std::is_same_v<PtdType<decltype(std::declval<Derived2>().clone())>,
                   Derived2>,
    "");
