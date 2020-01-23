// @H2_LICENSE_TEXT@

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Member.hpp"

using namespace h2::meta;
using namespace h2::meta::tlist;

using TList = TL<int, float, bool>;

// Testing Member
static_assert(MemberV<int, TList>(), "int in list");
static_assert(MemberV<bool, TList>(), "bool in list");
static_assert(MemberV<float, TList>(), "float in list");
static_assert(!MemberV<double, TList>(), "double not in list");
static_assert(!MemberV<int, Empty>(), "int not in empty list");
static_assert(!MemberV<TList, Empty>(), "TL not in empty list");
