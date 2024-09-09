////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include <stdexcept>

namespace h2
{
namespace factory
{
/** \class DefaultErrorPolicy
 *  \brief Handle unknown keys by throwing exceptions.
 */
template <typename IdType, class ObjectType>
struct DefaultErrorPolicy
{
  struct UnknownIDError : public std::exception
  {
    const char* what() const noexcept override
    {
      return "Unknown type identifier.";
    }
  };

  std::unique_ptr<ObjectType> handle_unknown_id(IdType const&) const
  {
    throw UnknownIDError();
  }
};  // struct DefaultErrorPolicy

}  // namespace factory
}  // namespace h2
