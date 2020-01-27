////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_META_PARTIAL_FUNCTIONS_MAKEFUNCTION_HPP_
#define H2_META_PARTIAL_FUNCTIONS_MAKEFUNCTION_HPP_

// TODO: Needs basic ValueList support

namespace h2
{
namespace meta
{
namespace pfunctions
{
/** @brief Function to produce N-ary functions for arbitrary N. */
template <template <typename...> class F, size_t N>
struct MakeNaryFunctionT;

/** @brief Function to produce N-ary functions for arbitrary N. */
template <template <typename...> class F, size_t N>
using MakeNaryFunction = Force<MakeNaryFunctionT<F, N>>;

/** @brief Make a unary metafunction. */
template <template <typename...> class F>
using MakeUnaryFunctionT = MakeNaryFunctionT<F, 1>;

/** @brief Make a unary metafunction. */
template <template <typename...> class F>
using MakeUnaryFunction = Force<MakeUnaryFunctionT<F>>;

/** @brief Make a binary metafunction. */
template <template <typename...> class F>
using MakeBinaryFunctionT = MakeNaryFunctionT<F, 2>;

/** @brief Make a binary metafunction. */
template <template <typename...> class F>
using MakeBinaryFunction = Force<MakeBinaryFunctionT<F>>;

/** @brief Make a ternary metafunction. */
template <template <typename...> class F>
using MakeTernaryFunctionT = MakeNaryFunctionT<F, 3>;

/** @brief Make a ternary metafunction. */
template <template <typename...> class F>
using MakeTernaryFunction = Force<MakeTernaryFunctionT<F>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace details
{
template <template <typename...> class F, typename ArgIdxs>
struct MakeNaryFunctionT_Impl;

template <template <typename...> class F, size_t... ArgIdxs>
struct MakeNaryFunctionT_Impl<F, ValueList<size_t, ArgIdxs...>>
{
    using type = F<PH<ArgIdxs>...>;
};
} // namespace details

template <template <typename...> class F, size_t N>
struct MakeNaryFunctionT
    : details::MakeNaryFunctionT_Impl<F, valuelist::MakeSizeTList<N>>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace pfunctions
} // namespace meta
} // namespace h2
#endif // H2_META_PARTIAL_FUNCTIONS_MAKEFUNCTION_HPP_
