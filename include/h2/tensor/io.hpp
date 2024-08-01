////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Printing and file I/O for tensors and distributed tensors.
 */

#include <ostream>
#include "tensor_types.hpp"

#include "h2/tensor/tensor.hpp"
#include "h2/tensor/copy.hpp"


namespace h2
{

/** Write tensor to the given stream. */
template <typename T>
inline std::ostream& print(std::ostream& os, const Tensor<T>& tensor)
{
  if (tensor.is_empty())
  {
    os << "[]";
    return os;
  }
  auto cpu_view = make_accessible_on_device(tensor, Device::CPU);
  // Ensure any copy is completed.
  cpu_view->get_stream().wait_for(tensor.get_stream());

  std::size_t indent = 0;  // Number of spaces to prepend.

  // Note: We traverse in a row-major order because that makes printing
  // nicer.
  ScalarIndexTuple coord{TuplePad<ScalarIndexTuple>(tensor.ndim(), 0)};
  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    // Print opening braces except for inner-most dim.
    for (; indent < coord.size() - 1; ++indent)
    {
      os << std::string(indent, ' ') << "[\n";
    }
    // Print inner-most opening brace.
    if (coord.back() == 0)
    {
      os << std::string(indent, ' ') << "[";
    }
    os << *(cpu_view->const_get(coord));
    // Close inner-most brace.
    if (coord.back() == tensor.shape().back() - 1)
    {
      os << "]";
      if (tensor.ndim() > 1)
      {
        os << "\n";
      }
    }
    else
    {
      os << ", ";
    }
    // Advance coordinates.
    coord.back() += 1;
    for (typename ScalarIndexTuple::size_type dim = coord.size() - 1; dim >= 0;
         --dim)
    {
      if (coord[dim] == tensor.shape(dim))
      {
        coord[dim] = 0;
        if (dim > 0) {
          coord[dim - 1] += 1;
        }
        // Close braces except for the inner-most dimension.
        if (dim < coord.size() - 1)
        {
          --indent;
          os << std::string(indent, ' ') << "]";
          // There is a newline unless this is the first dimension.
          if (dim > 0)
          {
            os << "\n";
          }
        }
      }
    }
  }

  return os;
}



}
