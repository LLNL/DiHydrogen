////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/partial_functions/MakeFunction.hpp"

using namespace h2::meta;
using namespace h2::meta::pfunctions;

namespace
{
template <typename T>
struct F1;

template <typename T, typename T>
struct F2;

template <typename T, typename T, typename T>
struct F3;

}  // namespace

static_assert(EqV<MakeNaryFunction<F1, 1>, F1<_1>>(), "Make a unary function.");
static_assert(EqV<MakeNaryFunction<F2, 2>, F2<_1, _2>>(),
              "Make a binary function.");
static_assert(EqV<MakeNaryFunction<F3, 3>, F3<_1, _2, _3>>(),
              "Make a ternary function.");

static_assert(EqV<MakeUnaryFunction<F1>, F1<_1>>(), "Make a unary function.");
static_assert(EqV<MakeBinaryFunction<F2>, F2<_1, _2>>(),
              "Make a binary function.");
static_assert(EqV<MakeTernaryFunction<F3>, F3<_1, _2, _3>>(),
              "Make a ternary function.");

namespace
{
template <typename T>
using F_Int = F2<T, int>;

template <typename T>
using FInt_ = F2<int, T>;

template <typename T>
using F_IntFloat = F3<T, int, float>;

template <typename T>
using FInt_Float = F3<int, T, float>;

template <typename T>
using FIntFloat_ = F3<int, float, T>;

template <typename T, typename U>
using FInt__ = F3<int, T, U>;

template <typename T, typename U>
using F_Int_ = F3<T, int, U>;

template <typename T, typename U>
using F__Int = F3<T, U, int>;

template <typename T, typename U>
using FRev = F2<U, T>;

}  // namespace

static_assert(EqV<MakeUnaryFunction<F_Int>, F2<_1, int>>(),
              "Make a unary function.");
static_assert(EqV<MakeUnaryFunction<FInt_>, F2<int, _1>>(),
              "Make a unary function.");
static_assert(EqV<MakeUnaryFunction<F_IntFloat>, F3<_1, int, float>>(),
              "Make a unary function.");
static_assert(EqV<MakeUnaryFunction<FInt_Float>, F3<int, _1, float>>(),
              "Make a unary function.");
static_assert(EqV<MakeUnaryFunction<FIntFloat_>, F3<int, float, _1>>(),
              "Make a unary function.");

static_assert(EqV<MakeUnaryFunction<F__Int>, F3<_1, _2, int>>(),
              "Make a binary function.");
static_assert(EqV<MakeUnaryFunction<F_Int_>, F3<_1, int, _2>>(),
              "Make a binary function.");
static_assert(EqV<MakeUnaryFunction<FInt__>, F3<int, _1, _2>>(),
              "Make a binary function.");
static_assert(EqV<MakeUnaryFunction<FRev>, F2<_2, _1>>(),
              "Make a binary function.");
