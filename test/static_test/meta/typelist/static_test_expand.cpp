// @H2_LICENSE_TEXT@

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Expand.hpp"

using namespace h2::meta;

template <typename T>
struct X;

using ResultTList1 = TL<X<char>>;
using ResultTList2 = TL<X<char>, X<int>, X<float>>;

static_assert(
    EqV<tlist::Expand<X>, tlist::Nil>(), "Expand no typelist returns nil.");
static_assert(
    EqV<tlist::Expand<X, char>, ResultTList1>(), "Expand single type.");
static_assert(
    EqV<tlist::Expand<X, char, int, float>, ResultTList2>(),
    "Expand over multiple types.");
