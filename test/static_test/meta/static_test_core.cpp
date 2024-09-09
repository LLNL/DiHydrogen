////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"

using namespace h2::meta;

// True/false type
static_assert(TrueType::value, "TrueType is true.");
static_assert(!FalseType::value, "FalseType is false.");

// Eq
static_assert(EqVT<int, int>::value, "EqVT is ok, true case.");
static_assert(!EqVT<int, float>::value, "EqVT is ok, false case.");

static_assert(EqV<char, char>(), "EqV() is ok, true case.");
static_assert(!EqV<char, double>(), "EqV() is ok, false case.");

static_assert(Eq<long, long>, "Eq is ok, true case.");
static_assert(!Eq<short, long>, "Eq is ok, false case.");

// Force
namespace static_test_core
{
struct Test
{
  using type = int;
};

static_assert(EqV<Force<Test>, int>(), "Force works.");
static_assert(!EqV<Force<Test>, long>(), "Force works.");

}  // namespace static_test_core
