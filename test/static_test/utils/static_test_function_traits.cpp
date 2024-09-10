////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/utils/function_traits.hpp"

#include <type_traits>

using namespace h2;

int nullary()
{
  return 42;
}
int unary(float)
{
  return 42;
}
void unary_v(float)
{}
int binary(float, double)
{
  return 42;
}
[[maybe_unused]] auto unary_lambda = [](float) -> int {
  return 42;
};

struct Test
{
  int nullary() { return 42; }
  int unary(float) { return 42; }
};

struct NullaryFunctor
{
  int operator()() { return 42; }
};
struct UnaryFunctor
{
  int operator()(float) { return 42; }
};

static_assert(std::is_same_v<FunctionTraits<decltype(nullary)>::RetT, int>);
static_assert(FunctionTraits<decltype(nullary)>::has_return);
static_assert(
  std::is_same_v<FunctionTraits<decltype(nullary)>::FuncT, decltype(nullary)>);
static_assert(FunctionTraits<decltype(nullary)>::arity == 0);

static_assert(std::is_same_v<FunctionTraits<decltype(unary)>::RetT, int>);
static_assert(FunctionTraits<decltype(unary)>::has_return);
static_assert(FunctionTraits<decltype(unary)>::arity == 1);
static_assert(std::is_same_v<FunctionTraits<decltype(unary)>::arg<0>, float>);

static_assert(std::is_same_v<FunctionTraits<decltype(unary_v)>::RetT, void>);
static_assert(!FunctionTraits<decltype(unary_v)>::has_return);

static_assert(FunctionTraits<decltype(binary)>::arity == 2);
static_assert(std::is_same_v<FunctionTraits<decltype(binary)>::arg<0>, float>);
static_assert(std::is_same_v<FunctionTraits<decltype(binary)>::arg<1>, double>);

static_assert(FunctionTraits<decltype(&unary)>::arity == 1);

static_assert(FunctionTraits<decltype(unary_lambda)>::arity == 1);
static_assert(
  std::is_same_v<FunctionTraits<decltype(unary_lambda)>::RetT, int>);
static_assert(FunctionTraits<decltype(unary_lambda)>::has_return);
static_assert(
  std::is_same_v<FunctionTraits<decltype(unary_lambda)>::arg<0>, float>);

static_assert(FunctionTraits<decltype(&Test::nullary)>::arity == 0);
static_assert(
  std::is_same_v<FunctionTraits<decltype(&Test::nullary)>::RetT, int>);
static_assert(FunctionTraits<decltype(&Test::nullary)>::has_return);
static_assert(
  std::is_same_v<FunctionTraits<decltype(&Test::nullary)>::ClassT, Test>);
static_assert(FunctionTraits<decltype(&Test::unary)>::arity == 1);
static_assert(
  std::is_same_v<FunctionTraits<decltype(&Test::unary)>::arg<0>, float>);
static_assert(
  std::is_same_v<FunctionTraits<decltype(&Test::unary)>::ClassT, Test>);

static_assert(FunctionTraits<NullaryFunctor>::arity == 0);
static_assert(std::is_same_v<FunctionTraits<NullaryFunctor>::RetT, int>);
static_assert(FunctionTraits<NullaryFunctor>::has_return);

static_assert(FunctionTraits<UnaryFunctor>::arity == 1);
static_assert(std::is_same_v<FunctionTraits<UnaryFunctor>::arg<0>, float>);

static_assert(FunctionTraits<std::function<int(float)>>::arity == 1);
static_assert(
  std::is_same_v<FunctionTraits<std::function<int(float)>>::RetT, int>);
static_assert(FunctionTraits<std::function<int(float)>>::has_return);
static_assert(
  std::is_same_v<FunctionTraits<std::function<int(float)>>::arg<0>, float>);
