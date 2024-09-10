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
make_tensor(TypeInfo const& tinfo,
            Device device,
            ShapeTuple const& shape,
            DimensionTypeTuple const& dim_types,
            StrideTuple const& strides = {},
            TensorAllocationStrategy alloc_type = StrictAlloc,
            std::optional<ComputeStream> const stream = std::nullopt);

/** Create a view of tensor. */
std::unique_ptr<BaseTensor> view(BaseTensor& tensor);

/** Create a subview of tensor. */
std::unique_ptr<BaseTensor> view(BaseTensor& tensor,
                                 IndexRangeTuple const& coords);

/** Create a constant view of tensor. */
std::unique_ptr<BaseTensor> view(BaseTensor const& tensor);

/** Create a constant subview of tensor. */
std::unique_ptr<BaseTensor> view(BaseTensor const& tensor,
                                 IndexRangeTuple const& coords);

/** Create a constant view of tensor. */
std::unique_ptr<BaseTensor> const_view(BaseTensor const& tensor);

/** Create a constant subview of tensor. */
std::unique_ptr<BaseTensor> const_view(BaseTensor const& tensor,
                                       IndexRangeTuple const& coords);

}  // namespace base
}  // namespace h2
