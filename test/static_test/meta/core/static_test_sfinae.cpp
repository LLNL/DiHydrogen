////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/core/Eq.hpp"
#include "h2/meta/core/SFINAE.hpp"

#include <utility>

using namespace h2::meta;

namespace
{
template <typename T>
struct Q
{
    template <typename U, EnableWhen<EqV<T, U>(), int> = 0>
    static float F(U const&);

    template <typename U, EnableUnless<EqV<T, U>(), int> = 0>
    static double F(U const&);
};
} // namespace

static_assert(SubstitutionSuccess<int>::value, "SubstitutionSuccess is true.");
static_assert(
    SubstitutionSuccess<float>::value, "SubstitutionSuccess is true.");
static_assert(
    SubstitutionSuccess<Q<int>>::value, "SubstitutionSuccess is true.");
static_assert(
    !SubstitutionSuccess<SubstitutionFailure>::value,
    "SubstitutionSuccess of SubstitutionFailure is false.");

static_assert(
    EqV<EnableIf<true, int>, int>(),
    "EnableIf returns a type when its parameter is true.");
static_assert(
    EqV<EnableWhen<true, char>, char>(),
    "EnableWhen returns a type when its parameter is true.");
static_assert(
    EqV<EnableUnless<false, float>, float>(),
    "EnableUnless returns a type when its parameter is false.");

static_assert(
    EqV<decltype(Q<int>::F(std::declval<int>())), float>(), "SFINAE works.");
static_assert(
    EqV<decltype(Q<int>::F(std::declval<float>())), double>(), "SFINAE works.");
