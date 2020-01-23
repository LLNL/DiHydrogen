// @H2_LICENSE_TEXT@

#include "h2/meta/core/Invocable.hpp"

using namespace h2::meta;

namespace
{
struct X;

template <typename T>
struct Q
{
    void operator()(int, float);
};

}// namespace <anon>

// Other function
extern void f(double, double);

// Functor invocability
static_assert(IsInvocableV<Q<int>, int, float>(),
              "Invocable structs are invocable.");
static_assert(IsInvocableV<Q<int>, int, int>(),
              "Invocable structs are invocable through implicit conversion.");

static_assert(!IsInvocableV<Q<int>, int>(),
              "Not enough arguments.");
static_assert(!IsInvocableV<Q<int>, int, X>(),
              "No implicit conversions.");
static_assert(!IsInvocableV<Q<int>, int, float, char, X>(),
              "Too many arguments.");

// Function invocability
static_assert(IsInvocableV<decltype(f), double, double>(),
              "Functions are invocable with exact type matches.");
static_assert(IsInvocableV<decltype(f), float, float>(),
              "Functions are invocable through implicit conversions.");
static_assert(!IsInvocableV<decltype(f), X, float>(),
              "No implicit conversions."); // implicit conversions
