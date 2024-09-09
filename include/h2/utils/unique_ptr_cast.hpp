////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Utilities for working with `std::unique_ptr`s.
 */

#include <memory>
#include <type_traits>

namespace h2
{

/**
 * Cast a unique_ptr to a base class to a unique_ptr to a derived class
 * of the base.
 *
 * The input unique_pointer will be invalidated.
 */
template <typename DerivedT, typename BaseT>
std::unique_ptr<DerivedT> downcast_uptr(std::unique_ptr<BaseT>& p)
{
  static_assert(
    std::is_base_of_v<BaseT, DerivedT>,
    "Cannot cast a unique_ptr from a base class to a non-derived class");
  return std::unique_ptr<DerivedT>(static_cast<DerivedT*>(p.release()));
}

template <typename DerivedT, typename BaseT>
std::unique_ptr<const DerivedT> downcast_uptr(std::unique_ptr<const BaseT>& p)
{
  static_assert(
    std::is_base_of_v<BaseT, DerivedT>,
    "Cannot cast a unique_ptr from a base class to a non-derived class");
  return std::unique_ptr<const DerivedT>(
    static_cast<const DerivedT*>(p.release()));
}

}  // namespace h2
