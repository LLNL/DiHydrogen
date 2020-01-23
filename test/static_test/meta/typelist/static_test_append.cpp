// @H2_LICENSE_TEXT@

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Append.hpp"

using namespace h2::meta;

using TList1 = TL<char, short, int>;
using TList2 = TL<unsigned char, unsigned short, unsigned int>;
using TList3 = TL<float, double, long double>;

using ResultTList12 =
    TL<char, short, int, unsigned char, unsigned short, unsigned int>;
using ResultTList123 =
    TL<char,
       short,
       int,
       unsigned char,
       unsigned short,
       unsigned int,
       float,
       double,
       long double>;

// Testing Append
static_assert(
    EqV<tlist::Append<tlist::Empty, tlist::Empty>, tlist::Empty>(),
    "tlist::Append empty and empty.");
static_assert(
    EqV<tlist::Append<tlist::Empty, TL<int>>, TL<int>>(),
    "tlist::Append empty and nonempty.");
static_assert(
    EqV<tlist::Append<TL<int>, tlist::Empty>, TL<int>>(),
    "tlist::Append nonempty and empty.");

// Two
static_assert(
    EqV<tlist::Append<TL<int>, TL<float>>, TL<int, float>>(),
    "tlist::Append two nonempty lists");
static_assert(
    EqV<tlist::Append<TList1, TList2>, ResultTList12>(),
    "tlist::Append two nonempty lists");

// Three lists
static_assert(
    EqV<tlist::Append<tlist::Empty, tlist::Empty, tlist::Empty>,
        tlist::Empty>(),
    "tlist::Append empty, empty, and empty.");
static_assert(
    EqV<tlist::Append<TList1, TList2, TList3>, ResultTList123>(),
    "tlist::Append three nonempty lists");
static_assert(
    EqV<tlist::Append<tlist::Empty, TList1, TList2, TList3>, ResultTList123>(),
    "tlist::Append three nonempty lists with empty (0)");
static_assert(
    EqV<tlist::Append<TList1, tlist::Empty, TList2, TList3>, ResultTList123>(),
    "tlist::Append three nonempty lists with empty (1)");
static_assert(
    EqV<tlist::Append<TList1, TList2, tlist::Empty, TList3>, ResultTList123>(),
    "tlist::Append three nonempty lists with empty (2)");
static_assert(
    EqV<tlist::Append<TList1, TList2, TList3, tlist::Empty>, ResultTList123>(),
    "tlist::Append three nonempty lists with empty (3)");
