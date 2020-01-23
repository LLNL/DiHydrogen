// @H2_LICENSE_TEXT@

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Remove.hpp"

using namespace h2::meta;
using namespace h2::meta::tlist;

using TList = TL<int, float, double, int, float, double>;

// Testing Remove
static_assert(EqV<
              Remove<TList, int>,
              TL<float, double, int, float, double>>(),
              "Remove first entry.");
static_assert(EqV<
              Remove<TList, float>,
              TL<int, double, int, float, double>>(),
              "Remove second entry.");
static_assert(EqV<
              Remove<TList, double>,
              TL<int, float, int, float, double>>(),
              "Remove third entry.");
static_assert(EqV<Remove<TList, bool>, TList>(),
              "Remove nonexistent type.");
static_assert(EqV<Remove<Empty, bool>, Empty>(),
              "Remove nonexistent type from empty list.");
