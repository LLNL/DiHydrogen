// @H2_LICENSE_TEXT@

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

#ifndef H2_NO_CPP17
static_assert(Eq<long, long>, "Eq is ok, true case.");
static_assert(!Eq<short, long>, "Eq is ok, false case.");
#endif // H2_NO_CPP17

// Force
namespace static_test_core
{
struct Test
{
    using type = int;
};

static_assert(EqV<Force<Test>, int>(), "Force works.");
static_assert(!EqV<Force<Test>, long>(), "Force works.");

}// namespace static_test_core
