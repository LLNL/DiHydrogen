// @H2_LICENSE_TEXT@

#include "h2/meta/core/Eq.hpp"
#include "h2/meta/core/Lazy.hpp"

using namespace h2::meta;

static_assert(
    EqV<Force<Susp<int>>, int>(), "Force returns the type held in Susp.");
