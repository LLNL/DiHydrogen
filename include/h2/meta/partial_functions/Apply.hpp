// @H2_LICENSE_TEXT@

#ifndef H2_META_PARTIAL_FUNCTIONS_APPLY_HPP_
#define H2_META_PARTIAL_FUNCTIONS_APPLY_HPP_

#include "Placeholders.hpp"
#include "h2/meta/Core.hpp"
#include "h2/meta/TypeList.hpp"

namespace h2
{
namespace meta
{
namespace pfunctions
{
/** @brief Replace placeholders with formal arguments.
 *
 *  The given function may be fully or partially applied.
 *
 *  @tparam F The function to which placeholder replacements should be
 *          applied.
 *  @tparam Args A TypeList of arguments to apply agains the
 *          placeholder arguments to F.
 */
template <typename F, typename Args>
struct ApplyT;

/** @brief Replace placeholders with formal arguments.
 *
 *  The given function may be fully or partially applied.
 *
 *  @tparam F The function to which placeholder replacements should be
 *          applied.
 *  @tparam Args A TypeList of arguments to apply agains the
 *          placeholder arguments to F.
 */
template <typename F, typename Args>
using Apply = Force<ApplyT<F, Args>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// ApplyT implementation
template <template <typename...> class F, typename... Params, typename... Args>
struct ApplyT<F<Params...>, TL<Args...>>
{
    using type = F<placeholders::PHReplace<Params, Args...>...>;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace pfunctions
} // namespace meta
} // namespace h2
#endif // H2_META_PARTIAL_FUNCTIONS_APPLY_HPP_
