// @H2_LICENSE_TEXT@

#ifndef H2_META_CORE_IFTHENELSE_HPP_
#define H2_META_CORE_IFTHENELSE_HPP_

#include "Lazy.hpp"

namespace h2
{
namespace meta
{

template <bool B, typename T, typename F>
struct IfThenElseT;

template <bool B, typename T, typename F>
using IfThenElse = Force<IfThenElseT<B, T, F>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <bool B, typename T, typename F>
struct IfThenElseT
{
    using type = F;
};

template <typename T, typename F>
struct IfThenElseT<true, T, F>
{
    using type = T;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

}// namespace meta
}// namespace h2
#endif // H2_META_CORE_IFTHENELSE_HPP_
