////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_META_PARTIAL_FUNCTIONS_PLACEHOLDERS_HPP_
#define H2_META_PARTIAL_FUNCTIONS_PLACEHOLDERS_HPP_

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/At.hpp"

namespace h2
{
namespace meta
{
namespace pfunctions
{
namespace placeholders
{
/** @brief A placeholder for an argument to a type function.
 *
 *  These are placeholder arguments used in defining functions. This
 *  is similar to `std::bind`. These may replace any formal argument and
 *  may be used in any order.
 */
template <unsigned long Idx>
struct Placeholder;

/** @brief A generic first argument. */
using _1 = Placeholder<0>;
/** @brief A generic second argument. */
using _2 = Placeholder<1>;
/** @brief A generic third argument. */
using _3 = Placeholder<2>;
/** @brief A generic fourth argument. */
using _4 = Placeholder<3>;
/** @brief A generic fifth argument. */
using _5 = Placeholder<4>;
/** @brief A generic sixth argument. */
using _6 = Placeholder<5>;
/** @brief A generic seventh argument. */
using _7 = Placeholder<6>;
/** @brief A generic eighth argument. */
using _8 = Placeholder<7>;
/** @brief A generic ninth argument. */
using _9 = Placeholder<8>;

/** @brief Replace placeholders with real types.
 *  @tparam CandidatePH The type to test as a placeholder.
 *  @tparam Replacements The formal arguments to choose from.
 */
template <typename CandidatePH, typename... Replacements>
struct PHReplaceT;

/** @brief Replace placeholders with real types.
 *  @tparam CandidatePH The type to test as a placeholder.
 *  @tparam Replacements The formal arguments to choose from.
 */
template <typename CandidatePH, typename... Replacements>
using PHReplace = Force<PHReplaceT<CandidatePH, Replacements...>>;

} // namespace placeholders
} // namespace pfunctions

/** @brief A placeholder wrapper for use in the meta namespace. */
template <unsigned long Idx>
using PH = pfunctions::placeholders::Placeholder<Idx>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace pfunctions
{
namespace placeholders
{
// Degenerate case: The candidate is not actually a placeholder
template <typename PlainOldType, typename... Replacements>
struct PHReplaceT
{
    using type = PlainOldType;
};

// Short cut the first few placeholders
template <typename Replacement, typename... Others>
struct PHReplaceT<_1, Replacement, Others...>
{
    using type = Replacement;
};

template <typename S, typename Replacement, typename... Others>
struct PHReplaceT<_2, S, Replacement, Others...>
{
    using type = Replacement;
};

template <typename R, typename S, typename Replacement, typename... Others>
struct PHReplaceT<_3, R, S, Replacement, Others...>
{
    using type = Replacement;
};

// General placeholder case
template <unsigned long Idx, typename... Replacements>
struct PHReplaceT<Placeholder<Idx>, Replacements...>
{
private:
    static constexpr unsigned long num_args_ = sizeof...(Replacements);
    static constexpr bool do_arg_replace_ = (Idx < num_args_);

public:
    using type = IfThenElse<do_arg_replace_,
                            tlist::At<TL<Replacements...>, Idx>,
                            Placeholder<Idx - num_args_>>;
};

} // namespace placeholders

// Inject these symbols into the pfunctions namespace
using placeholders::_1;
using placeholders::_2;
using placeholders::_3;
using placeholders::_4;
using placeholders::_5;
using placeholders::_6;
using placeholders::_7;
using placeholders::_8;
using placeholders::_9;

} // namespace pfunctions
#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace meta
} // namespace h2
#endif // H2_META_PARTIAL_FUNCTIONS_PLACEHOLDERS_HPP_
