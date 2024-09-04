////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_META_CORE_LAZY_HPP_
#define H2_META_CORE_LAZY_HPP_

namespace h2
{
namespace meta
{
/** @brief Suspend a given type. */
template <typename T>
struct Susp
{
    using type = T;
};

/** @brief Extract the internal type from a suspended type. */
template <typename SuspT>
using Force = typename SuspT::type;

} // namespace meta
} // namespace h2
#endif // H2_META_CORE_LAZY_HPP_
