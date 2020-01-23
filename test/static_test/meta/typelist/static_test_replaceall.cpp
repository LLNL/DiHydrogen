// @H2_LICENSE_TEXT@

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/ReplaceAll.hpp"

using namespace h2::meta;
using namespace h2::meta::tlist;

// Testing ReplaceAll
static_assert(EqV<ReplaceAll<Empty, int, float>, Empty>(),
              "Replacing in an empty list has no effect.");
static_assert(EqV<
              ReplaceAll<TL<int, int, int>, int, float>,
              TL<float, float, float>>(),
              "ReplaceAll list of the same type.");

using TList = TL<int, float, double, int, float, double>;
static_assert(EqV<
              ReplaceAll<TList, double, char>,
              TL<int, float, char, int, float, char>>(),
              "ReplaceAll in an interesting list.");

static_assert(EqV<ReplaceAll<TList, unsigned, char>, TList>(),
              "ReplaceAll a nonexistent type has no effect.");
