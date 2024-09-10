////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Various utilities for distributed tensors.
 */

#include "h2/tensor/dist_types.hpp"
#include "h2/tensor/proc_grid.hpp"
#include "h2/tensor/tensor_types.hpp"
#include "h2/tensor/tensor_utils.hpp"

#include <algorithm>

#include "tensor_types.hpp"

namespace h2
{
namespace internal
{

// Utilities for converting sizes and indices between global and local.
// All of these include versions that operate on a dimension in
// isolation and a version that operates on a full shape/coordinate set
// or similar. Be aware that per-dimension results cannot be naively
// composed: Single distributions may result in some ranks having no
// data, despite their other distributions.

/**
 * Get the local size of a dimension.
 *
 * @warning This treats a dimension in isolation.
 */
template <Distribution Dist>
inline typename ShapeTuple::type
get_dim_local_size(typename ShapeTuple::type dim_size,
                   typename ShapeTuple::type grid_dim_size,
                   RankType grid_dim_rank,
                   bool is_root);

template <>
inline typename ShapeTuple::type
get_dim_local_size<Distribution::Block>(typename ShapeTuple::type dim_size,
                                        typename ShapeTuple::type grid_dim_size,
                                        RankType grid_dim_rank,
                                        bool /*is_root*/)
{
  typename ShapeTuple::type const remainder = dim_size % grid_dim_size;
  return (dim_size / grid_dim_size)
         + (grid_dim_rank < remainder ? ShapeTuple::type{1}
                                      : ShapeTuple::type{0});
}

template <>
inline typename ShapeTuple::type get_dim_local_size<Distribution::Replicated>(
  typename ShapeTuple::type dim_size,
  typename ShapeTuple::type /*grid_dim_size*/,
  RankType /*grid_dim_rank*/,
  bool /*is_root*/)
{
  return dim_size;
}

template <>
inline typename ShapeTuple::type get_dim_local_size<Distribution::Single>(
  typename ShapeTuple::type dim_size,
  typename ShapeTuple::type /*grid_dim_size*/,
  RankType /*grid_dim_rank*/,
  bool is_root)
{
  return is_root ? dim_size : ShapeTuple::type{0};
}

/**
 * Return the local size of a dimension based on the processor grid
 * and distribution.
 */
inline typename ShapeTuple::type
get_dim_local_size(typename ShapeTuple::type dim_size,
                   typename ShapeTuple::size_type dim,
                   ProcessorGrid const& proc_grid,
                   Distribution dist,
                   RankType grid_rank)
{
  H2_ASSERT_DEBUG(grid_rank < proc_grid.size(),
                  "Invalid grid rank ",
                  grid_rank,
                  " (max grid rank ",
                  proc_grid.size(),
                  ")");
  ShapeTuple::type const grid_dim_size = proc_grid.shape(dim);
  RankType const grid_dim_rank = proc_grid.get_dimension_rank(dim, grid_rank);
  switch (dist)
  {
  case Distribution::Block:
    return get_dim_local_size<Distribution::Block>(
      dim_size, grid_dim_size, grid_dim_rank, false);
  case Distribution::Replicated:
    return get_dim_local_size<Distribution::Replicated>(
      dim_size, grid_dim_size, grid_dim_rank, false);
  case Distribution::Single:
    return get_dim_local_size<Distribution::Single>(
      dim_size, grid_dim_size, grid_dim_rank, grid_dim_rank == 0);
  default: H2_ASSERT_ALWAYS(false, "Invalid distribution ", dist);
  }
}

inline typename ShapeTuple::type
get_dim_local_size(typename ShapeTuple::type dim_size,
                   typename ShapeTuple::size_type dim,
                   ProcessorGrid const& proc_grid,
                   Distribution dist)
{
  return get_dim_local_size(dim_size, dim, proc_grid, dist, proc_grid.rank());
}

/**
 * Get the local shape given a global shape, processor grid, and
 * distributions.
 *
 * @note This is only correct for tensors that are not views (or for
 * which their local data is entirely present in the view).
 */
inline ShapeTuple get_local_shape(ShapeTuple shape,
                                  ProcessorGrid const& proc_grid,
                                  DistributionTypeTuple dist,
                                  RankType grid_rank)
{
  ShapeTuple local_shape(TuplePad<ShapeTuple>(shape.size(), 0));
  for (typename ShapeTuple::size_type dim = 0; dim < shape.size(); ++dim)
  {
    local_shape[dim] =
      get_dim_local_size(shape[dim], dim, proc_grid, dist[dim], grid_rank);
    if (local_shape[dim] == 0)
    {
      // No data, shape is empty.
      return ShapeTuple();
    }
  }
  return local_shape;
}

inline ShapeTuple get_local_shape(ShapeTuple shape,
                                  ProcessorGrid const& proc_grid,
                                  DistributionTypeTuple dist)
{
  return get_local_shape(shape, proc_grid, dist, proc_grid.rank());
}

/**
 * Get the indices of a dimension that are present on a given rank.
 *
 * @warning This treats a dimension in isolation.
 */
template <Distribution Dist>
inline IndexRange
get_dim_global_indices(typename ShapeTuple::type dim_size,
                       typename ShapeTuple::type grid_dim_size,
                       RankType grid_dim_rank,
                       bool is_root);

template <>
inline IndexRange get_dim_global_indices<Distribution::Block>(
  typename ShapeTuple::type dim_size,
  typename ShapeTuple::type grid_dim_size,
  RankType grid_dim_rank,
  bool /*is_root*/)
{
  ShapeTuple::type const remainder = dim_size % grid_dim_size;
  ShapeTuple::type const block_size = dim_size / grid_dim_size;
  ShapeTuple::type const start =
    block_size * grid_dim_rank
    + std::min(remainder, static_cast<ShapeTuple::type>(grid_dim_rank));
  // Handle case where some ranks have no indices.
  return (start >= dim_size)
           ? IndexRange()
           : IndexRange(start,
                        start + block_size
                          + (grid_dim_rank < remainder ? ShapeTuple::type{1}
                                                       : ShapeTuple::type{0}));
}

template <>
inline IndexRange get_dim_global_indices<Distribution::Replicated>(
  typename ShapeTuple::type dim_size,
  typename ShapeTuple::type /*grid_dim_size*/,
  RankType /*grid_dim_rank*/,
  bool /*is_root*/)
{
  return IndexRange(0, dim_size);
}

template <>
inline IndexRange get_dim_global_indices<Distribution::Single>(
  typename ShapeTuple::type dim_size,
  typename ShapeTuple::type /*grid_dim_size*/,
  RankType /*grid_dim_rank*/,
  bool is_root)
{
  return is_root ? IndexRange(0, dim_size) : IndexRange();
}

inline IndexRange get_dim_global_indices(typename ShapeTuple::type dim_size,
                                         typename ShapeTuple::size_type dim,
                                         ProcessorGrid const& proc_grid,
                                         Distribution dist,
                                         RankType grid_rank)
{
  H2_ASSERT_DEBUG(grid_rank < proc_grid.size(),
                  "Invalid grid rank ",
                  grid_rank,
                  " (max gird rank, ",
                  proc_grid.size(),
                  ")");
  ShapeTuple::type const grid_dim_size = proc_grid.shape(dim);
  RankType const grid_dim_rank = proc_grid.get_dimension_rank(dim, grid_rank);
  switch (dist)
  {
  case Distribution::Block:
    return get_dim_global_indices<Distribution::Block>(
      dim_size, grid_dim_size, grid_dim_rank, false);
  case Distribution::Replicated:
    return get_dim_global_indices<Distribution::Replicated>(
      dim_size, grid_dim_size, grid_dim_rank, false);
  case Distribution::Single:
    return get_dim_global_indices<Distribution::Single>(
      dim_size, grid_dim_size, grid_dim_rank, grid_dim_rank == 0);
  default: H2_ASSERT_ALWAYS(false, "Invalid distribution ", dist);
  }
}

inline IndexRange get_dim_global_indices(typename ShapeTuple::type dim_size,
                                         typename ShapeTuple::size_type dim,
                                         ProcessorGrid const& proc_grid,
                                         Distribution dist)
{
  return get_dim_global_indices(
    dim_size, dim, proc_grid, dist, proc_grid.rank());
}

/**
 * Get the indices present on a rank given a global shape,
 * processor grid, and distributions.
 */
inline IndexRangeTuple get_global_indices(ShapeTuple global_shape,
                                          ProcessorGrid const& proc_grid,
                                          DistributionTypeTuple dist,
                                          RankType grid_rank)
{
  IndexRangeTuple indices(TuplePad<IndexRangeTuple>(global_shape.size()));
  for (typename ShapeTuple::size_type dim = 0; dim < global_shape.size(); ++dim)
  {
    indices[dim] = get_dim_global_indices(
      global_shape[dim], dim, proc_grid, dist[dim], grid_rank);
    if (indices[dim].is_empty())
    {
      // No data, set all dimension indices to 0.
      return IndexRangeTuple(TuplePad<IndexRangeTuple>(global_shape.size()));
    }
  }
  return indices;
}

inline IndexRangeTuple get_global_indices(ShapeTuple global_shape,
                                          ProcessorGrid const& proc_grid,
                                          DistributionTypeTuple dist)
{
  return get_global_indices(global_shape, proc_grid, dist, proc_grid.rank());
}

/**
 * Convert a global index to a local index for a dimension.
 *
 * @warning This gives the associated local index for whatever rank the
 * indices are on, which may not be the caller.
 */
template <Distribution Dist>
inline DimType dim_global2local_index(typename ShapeTuple::type dim_size,
                                      typename ShapeTuple::type grid_dim_size,
                                      DimType global_index);

template <>
inline DimType dim_global2local_index<Distribution::Block>(
  typename ShapeTuple::type dim_size,
  typename ShapeTuple::type grid_dim_size,
  DimType global_index)
{
  ShapeTuple::type const remainder = dim_size % grid_dim_size;
  ShapeTuple::type const block_size = dim_size / grid_dim_size;
  if (global_index < (block_size + 1) * remainder)
  {
    return global_index % (block_size + 1);
  }
  else
  {
    return (global_index - ((block_size + 1) * remainder)) % block_size;
  }
}

template <>
inline DimType dim_global2local_index<Distribution::Replicated>(
  typename ShapeTuple::type /*dim_size*/,
  typename ShapeTuple::type /*grid_dim_size*/,
  DimType global_index)
{
  return global_index;
}

template <>
inline DimType dim_global2local_index<Distribution::Single>(
  typename ShapeTuple::type /*dim_size*/,
  typename ShapeTuple::type /*grid_dim_size*/,
  DimType global_index)
{
  return global_index;
}

inline DimType dim_global2local_index(typename ShapeTuple::type dim_size,
                                      typename ShapeTuple::size_type dim,
                                      ProcessorGrid const& proc_grid,
                                      Distribution dist,
                                      DimType global_index)
{
  H2_ASSERT_DEBUG(global_index < dim_size,
                  "Invalid global index ",
                  global_index,
                  " (max dimension size ",
                  dim_size,
                  ")");
  ShapeTuple::type const grid_dim_size = proc_grid.shape(dim);
  switch (dist)
  {
  case Distribution::Block:
    return dim_global2local_index<Distribution::Block>(
      dim_size, grid_dim_size, global_index);
  case Distribution::Replicated:
    return dim_global2local_index<Distribution::Replicated>(
      dim_size, grid_dim_size, global_index);
  case Distribution::Single:
    return dim_global2local_index<Distribution::Single>(
      dim_size, grid_dim_size, global_index);
  default: H2_ASSERT_ALWAYS(false, "Invalid distribution ", dist);
  }
}

inline ScalarIndexTuple global2local_index(ShapeTuple global_shape,
                                           ProcessorGrid const& proc_grid,
                                           DistributionTypeTuple dist,
                                           ScalarIndexTuple global_index)
{
  return map_index(global_index, [&](ScalarIndexTuple::size_type dim) {
    return dim_global2local_index(
      global_shape[dim], dim, proc_grid, dist[dim], global_index[dim]);
  });
}

inline IndexRangeTuple global2local_indices(ShapeTuple global_shape,
                                            ProcessorGrid const& proc_grid,
                                            DistributionTypeTuple dist,
                                            IndexRangeTuple global_indices)
{
  H2_ASSERT_DEBUG(!any_of(global_indices,
                          [](typename IndexRangeTuple::type const& c) {
                            return c.is_empty();
                          }),
                  "Empty index entries are not supported, got ",
                  global_indices);
  return map_index(global_indices, [&](IndexRangeTuple::size_type dim) {
    if (global_indices[dim].is_scalar())
    {
      return IndexRange(dim_global2local_index(global_shape[dim],
                                               dim,
                                               proc_grid,
                                               dist[dim],
                                               global_indices[dim].start()));
    }
    else
    {
      // This is a half-open range, hence the end index may not be one
      // that exists in the shape. We subtract one to work with a valid
      // index, and then add 1 to restore the original half-open range.
      return IndexRange(dim_global2local_index(global_shape[dim],
                                               dim,
                                               proc_grid,
                                               dist[dim],
                                               global_indices[dim].start()),
                        dim_global2local_index(global_shape[dim],
                                               dim,
                                               proc_grid,
                                               dist[dim],
                                               global_indices[dim].end() - 1)
                          + 1);
    }
  });
}

/** Get the rank that has a global index for a dimension. */
template <Distribution Dist>
inline RankType dim_global2rank(typename ShapeTuple::type dim_size,
                                typename ShapeTuple::type grid_dim_size,
                                DimType global_index);

template <>
inline RankType
dim_global2rank<Distribution::Block>(typename ShapeTuple::type dim_size,
                                     typename ShapeTuple::type grid_dim_size,
                                     DimType global_index)
{
  ShapeTuple::type const remainder = dim_size % grid_dim_size;
  ShapeTuple::type const block_size = dim_size / grid_dim_size;
  if (global_index < (block_size + 1) * remainder)
  {
    return global_index / (block_size + 1);
  }
  else
  {
    return (global_index - ((block_size + 1) * remainder)) / block_size
           + remainder;
  }
}

template <>
inline RankType dim_global2rank<Distribution::Replicated>(
  typename ShapeTuple::type /*dim_size*/,
  typename ShapeTuple::type /*grid_dim_size*/,
  DimType /*global_index*/)
{
  return 0;  // Data is present on all ranks in the grid dimension.
}

template <>
inline RankType dim_global2rank<Distribution::Single>(
  typename ShapeTuple::type /*dim_size*/,
  typename ShapeTuple::type /*grid_dim_size*/,
  DimType /*global_index*/)
{
  return 0;  // Data is always present on the root.
}

inline RankType dim_global2rank(typename ShapeTuple::type dim_size,
                                typename ShapeTuple::size_type dim,
                                ProcessorGrid const& proc_grid,
                                Distribution dist,
                                DimType global_index)
{
  H2_ASSERT_DEBUG(global_index < dim_size,
                  "Invalid global index ",
                  global_index,
                  " (max dimension size ",
                  dim_size,
                  ")");
  ShapeTuple::type const grid_dim_size = proc_grid.shape(dim);
  switch (dist)
  {
  case Distribution::Block:
    return dim_global2rank<Distribution::Block>(
      dim_size, grid_dim_size, global_index);
  case Distribution::Replicated:
    // Instead of the specialization, which pessimistically returns 0,
    // we just return the caller's dimension rank.
    return proc_grid.get_dimension_rank(dim);
  case Distribution::Single:
    return dim_global2rank<Distribution::Single>(
      dim_size, grid_dim_size, global_index);
  default: H2_ASSERT_ALWAYS(false, "Invalid distribution ", dist);
  }
}

inline RankType global2rank(ShapeTuple global_shape,
                            ProcessorGrid const& proc_grid,
                            DistributionTypeTuple dist,
                            ScalarIndexTuple global_index)
{
  ScalarIndexTuple grid_index =
    map_index(global_index, [&](ScalarIndexTuple::size_type dim) {
      return dim_global2rank(
        global_shape[dim], dim, proc_grid, dist[dim], global_index[dim]);
    });
  return proc_grid.rank(grid_index);
}

/**
 * Convert a dimension rank and local index to the associated global
 * dimension index.
 */
template <Distribution Dist>
inline DimType dim_local2global_index(typename ShapeTuple::type dim_size,
                                      typename ShapeTuple::type grid_dim_size,
                                      RankType grid_dim_rank,
                                      DimType local_index);

template <>
inline DimType dim_local2global_index<Distribution::Block>(
  typename ShapeTuple::type dim_size,
  typename ShapeTuple::type grid_dim_size,
  RankType grid_dim_rank,
  DimType local_index)
{
  ShapeTuple::type const remainder = dim_size % grid_dim_size;
  ShapeTuple::type const block_size = dim_size / grid_dim_size;
  if (grid_dim_rank < remainder)
  {
    return (grid_dim_rank * (block_size + 1)) + local_index;
  }
  else
  {
    return (remainder * (block_size + 1))
           + (block_size * (grid_dim_rank - remainder)) + local_index;
  }
}

template <>
inline DimType dim_local2global_index<Distribution::Replicated>(
  typename ShapeTuple::type /*dim_size*/,
  typename ShapeTuple::type /*grid_dim_size*/,
  RankType /*grid_dim_rank*/,
  DimType local_index)
{
  return local_index;
}

template <>
inline DimType dim_local2global_index<Distribution::Single>(
  typename ShapeTuple::type /*dim_size*/,
  typename ShapeTuple::type /*grid_dim_size*/,
  RankType /*grid_dim_rank*/,
  DimType local_index)
{
  return local_index;
}

inline DimType dim_local2global_index(typename ShapeTuple::type dim_size,
                                      typename ShapeTuple::size_type dim,
                                      ProcessorGrid const& proc_grid,
                                      Distribution dist,
                                      RankType grid_dim_rank,
                                      DimType local_index)
{
  H2_ASSERT_DEBUG(grid_dim_rank < proc_grid.shape(dim),
                  "Invalid grid dimension rank ",
                  grid_dim_rank,
                  " for dimension ",
                  dim,
                  " of grid ",
                  proc_grid.shape());
  H2_ASSERT_DEBUG(local_index < dim_size,
                  "Invalid local index ",
                  local_index,
                  " (max dimension size ",
                  dim_size,
                  ")");
  ShapeTuple::type const grid_dim_size = proc_grid.shape(dim);
  switch (dist)
  {
  case Distribution::Block:
    return dim_local2global_index<Distribution::Block>(
      dim_size, grid_dim_size, grid_dim_rank, local_index);
  case Distribution::Replicated:
    return dim_local2global_index<Distribution::Replicated>(
      dim_size, grid_dim_size, grid_dim_rank, local_index);
  case Distribution::Single:
    return dim_local2global_index<Distribution::Single>(
      dim_size, grid_dim_size, grid_dim_rank, local_index);
  default: H2_ASSERT_ALWAYS(false, "Invalid distribution ", dist);
  }
}

inline ScalarIndexTuple local2global_index(ShapeTuple global_shape,
                                           ProcessorGrid const& proc_grid,
                                           DistributionTypeTuple dist,
                                           RankType grid_rank,
                                           ScalarIndexTuple local_index)
{
  return map_index(local_index, [&](ScalarIndexTuple::size_type dim) {
    H2_ASSERT_DEBUG(local_index[dim] < get_dim_local_size(
                      global_shape[dim], dim, proc_grid, dist[dim], grid_rank),
                    "Invalid local index ",
                    local_index);
    return dim_local2global_index(global_shape[dim],
                                  dim,
                                  proc_grid,
                                  dist[dim],
                                  proc_grid.get_dimension_rank(dim, grid_rank),
                                  local_index[dim]);
  });
}

}  // namespace internal
}  // namespace h2
