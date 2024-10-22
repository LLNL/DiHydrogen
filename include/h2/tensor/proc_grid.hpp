////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Processor grids and associated types.
 */

#include "h2/tensor/dist_types.hpp"
#include "h2/tensor/tensor_types.hpp"
#include "h2/utils/As.hpp"
#include "h2/utils/Describable.hpp"
#include "h2/utils/Error.hpp"

#include <memory>

namespace h2
{

/**
 * Defines a grid of processes which a tensor may be distributed over.
 *
 * A processor grid consists of a communicator, which defines the
 * participating processors, and a shape, which defines the actual
 * process grid. It is an error if the size of the communicator does
 * not exactly match the number of processors required for the grid.
 *
 * Currently processors are mapped to the grid in a generalized column-
 * major order. (This may be configurable in the future.)
 *
 * Grids also provide a notion of a rank in each dimension.
 *
 * The underlying communicator will be duplicated.
 *
 * \note There is no attempt at "topology-aware" mapping here (e.g.,
 * `MPI_Cart_create`). Users should manually manage this on the input
 * communicator if desired.
 */
class ProcessorGrid final : public Describable
{
private:
  /** Strides on the grid. */
  using GridStrideTuple = NDimTuple<RankType>;

public:
  /**
   * Construct a processor grid of the given shape over the communicator.
   */
  ProcessorGrid(Comm const& comm_, ShapeTuple shape_)
  {
    H2_ASSERT_ALWAYS(comm_.size() == product<RankType>(shape_),
                     "Grid size (",
                     shape_,
                     ") must match communicator size (",
                     comm_.size(),
                     ")");
    grid_comm = std::make_shared<Comm>(comm_.get_mpi_handle());
    grid_shape = shape_;
    // Currently fix a column-major ordering.
    grid_strides = prefix_product<typename GridStrideTuple::type>(shape_);
  }

  /** Construct a null processor grid. */
  ProcessorGrid() { grid_comm = std::make_shared<Comm>(); }

  /** Get a reference to the underlying communicator. */
  Comm& comm() H2_NOEXCEPT { return *grid_comm; }

  /** Get a constant reference to the underlying communicator. */
  Comm const& comm() const H2_NOEXCEPT { return *grid_comm; }

  /** Return the shape of the grid. */
  ShapeTuple shape() const H2_NOEXCEPT { return grid_shape; }

  /** Return the size of a particular grid dimension. */
  typename ShapeTuple::type
  shape(typename ShapeTuple::size_type i) const H2_NOEXCEPT
  {
    return grid_shape[i];
  }

  /** Return the number of dimensions (i.e., the rank) of the grid. */
  typename ShapeTuple::size_type ndim() const H2_NOEXCEPT
  {
    return grid_shape.size();
  }

  /** Output a short description of the processor grid. */
  void short_describe(std::ostream& os) const final
  {
    os << "Grid";
    print_tuple(os, shape(), "(", ")", " x ");
  }

  /** Return the number of processors in the grid. */
  RankType size() const H2_NOEXCEPT { return grid_comm->size(); }

  /** Return the rank of the calling process in the grid. */
  RankType rank() const H2_NOEXCEPT { return grid_comm->rank(); }

  /** Return the rank of the process at a given grid coordinate. */
  RankType rank(ScalarIndexTuple coord) const H2_NOEXCEPT
  {
    return inner_product<RankType>(coord, grid_strides);
  }

  /** Return the coordinates on the grid of a given rank in the grid. */
  ScalarIndexTuple coords(RankType rank) const H2_NOEXCEPT
  {
    ScalarIndexTuple coord(TuplePad<ScalarIndexTuple>(ndim()));
    for (typename ShapeTuple::size_type i = 0; i < grid_shape.size(); ++i)
    {
      coord[i] = (rank / grid_strides[i]) % grid_shape[i];
    }
    return coord;
  }

  /** Return the coordinates on the grid of the calling process. */
  ScalarIndexTuple coords() const H2_NOEXCEPT { return coords(rank()); }

  /**
   * Return the rank in a particular dimension of a grid rank.
   *
   * The grid rank is the global rank of a process (i.e., what you get
   * from `rank`). The dimension rank is the index of the process in
   * the particular dimension. Essentially, this equivalent to:
   *
   * ```
   * coords(grid_rank)[i]
   * ```
   */
  RankType get_dimension_rank(typename ShapeTuple::size_type dim,
                              RankType grid_rank) const H2_NOEXCEPT
  {
    return (grid_rank / grid_strides[dim]) % grid_shape[dim];
  }

  /**
   * Return the rank in a particular dimension of the calling process.
   */
  RankType
  get_dimension_rank(typename ShapeTuple::size_type dim) const H2_NOEXCEPT
  {
    return get_dimension_rank(dim, rank());
  }

  /**
   * Return true if rank in the provided communicator (default is the
   * world) is part of this grid.
   */
  bool participating(RankType rank,
                     Comm const& comm = get_comm_world()) const H2_NOEXCEPT
  {
    MPI_Group this_group;
    MPI_Comm_group(grid_comm->get_mpi_handle(), &this_group);
    MPI_Group other_group;
    MPI_Comm_group(comm.get_mpi_handle(), &other_group);
    int in_rank = safe_as<int>(rank);
    int out_rank;
    MPI_Group_translate_ranks(other_group, 1, &in_rank, this_group, &out_rank);
    return out_rank != MPI_UNDEFINED;
  }

  /**
   * Return true if rank in the provided grid is part of this grid.
   */
  bool participating(RankType rank, ProcessorGrid const& grid) const H2_NOEXCEPT
  {
    return participating(rank, *grid.grid_comm);
  }

  /**
   * Return true if this grid is identical to the other grid.
   *
   * Two grids are identical if they have the same shape and use the
   * same underlying communicator.
   */
  bool is_identical_to(ProcessorGrid const& other) const H2_NOEXCEPT
  {
    return (grid_shape == other.grid_shape)
           && (grid_comm->get_mpi_handle()
               == other.grid_comm->get_mpi_handle());
  }

  /**
   * Return true if this grid is congruent to the other grid.
   *
   * Two grids are congruent if they have the same shape and the
   * underlying communicators consist of the same processes in the same
   * order (i.e., they are `MPI_CONGRUENT`).
   */
  bool is_congruent_to(ProcessorGrid const& other) const H2_NOEXCEPT
  {
    if (grid_shape != other.grid_shape)
    {
      return false;
    }
    // MPI_Comm_compare does not handle MPI_COMM_NULL.
    if (grid_comm->get_mpi_handle() == MPI_COMM_NULL
        && other.grid_comm->get_mpi_handle() == MPI_COMM_NULL)
    {
      return true;
    }
    int result;
    MPI_Comm_compare(
      grid_comm->get_mpi_handle(), other.grid_comm->get_mpi_handle(), &result);
    return result == MPI_IDENT || result == MPI_CONGRUENT;
  }

private:
  /** Underlying communicator for the grid. */
  std::shared_ptr<Comm> grid_comm;
  ShapeTuple grid_shape;        /**< Shape of the grid. */
  GridStrideTuple grid_strides; /**< Strides for computing indices. */
};

/**
 * Equality for processor grids.
 *
 * Two grids are equal when they have the same shape and the same
 * underlying communicator.
 */
inline bool operator==(ProcessorGrid const& grid1, ProcessorGrid const& grid2)
{
  return grid1.is_identical_to(grid2);
}

/** Inequality for processor grids. */
inline bool operator!=(ProcessorGrid const& grid1, ProcessorGrid const& grid2)
{
  return !grid1.is_identical_to(grid2);
}

}  // namespace h2
