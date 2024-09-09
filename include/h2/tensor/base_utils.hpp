////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Helper routines for working with Base(Dist)Tensors.
 *
 * These are intended to help smooth over components of the API where
 * runtime types are involved and we cannot easily use standard
 * polymorphism.
 *
 * These use H2's dynamic dispatch infrastructure and therefore only
 * support compute types.
 */

#include "h2/tensor/dist_tensor_base.hpp"
#include "h2/tensor/tensor_base.hpp"

namespace h2
{
namespace base
{

/**
 * Create a new Tensor with type given in tinfo and other arguments
 * as in the Tensor constructor.
 */
std::unique_ptr<BaseTensor>
make_tensor(const TypeInfo& tinfo,
            Device device,
            const ShapeTuple& shape,
            const DimensionTypeTuple& dim_types,
            const StrideTuple& strides = {},
            TensorAllocationStrategy alloc_type = StrictAlloc,
            const std::optional<ComputeStream> stream = std::nullopt);

/** Create a view of tensor. */
std::unique_ptr<BaseTensor> view(BaseTensor& tensor);

/** Create a subview of tensor. */
std::unique_ptr<BaseTensor> view(BaseTensor& tensor,
                                 const IndexRangeTuple& coords);

/** Create a constant view of tensor. */
std::unique_ptr<BaseTensor> view(const BaseTensor& tensor);

/** Create a constant subview of tensor. */
std::unique_ptr<BaseTensor> view(const BaseTensor& tensor,
                                 const IndexRangeTuple& coords);

/** Create a constant view of tensor. */
std::unique_ptr<BaseTensor> const_view(const BaseTensor& tensor);

/** Create a constant subview of tensor. */
std::unique_ptr<BaseTensor> const_view(const BaseTensor& tensor,
                                       const IndexRangeTuple& coords);

}  // namespace base
}  // namespace h2
