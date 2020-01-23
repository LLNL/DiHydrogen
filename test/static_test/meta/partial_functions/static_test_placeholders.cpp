// @H2_LICENSE_TEXT@

#include "h2/meta/partial_functions/Placeholders.hpp"

#include "h2/meta/Core.hpp"

using namespace h2::meta;
using namespace h2::meta::pfunctions::placeholders;

static_assert(EqV<PHReplace<int, int, float, double>, int>(),
              "No replacement for non-placeholder.");
static_assert(EqV<PHReplace<_1, int, float, double>, int>(),
              "Replacement for first type.");
static_assert(EqV<PHReplace<_2, int, float, double>, float>(),
              "Replacement for second type.");
static_assert(EqV<PHReplace<_3, int, float, double>, double>(),
              "Replacement for third type.");
static_assert(EqV<PHReplace<_4, int, float, double>, _1>(),
              "Replacement out of given range reduces the index.");
static_assert(EqV<PHReplace<_6, int, float, double>, _3>(),
              "Replacement out of given range reduces the index.");
