////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2/tensor/dist_tensor.hpp"
#include "h2/tensor/proc_grid.hpp"
#include "h2/tensor/tensor_types.hpp"

#include <El.hpp>

#include <memory>
#include <type_traits>

#include "interop_utils.hpp"

namespace h2
{
namespace internal
{

// Note that all new tensors inherit the SyncInfo associated with the
// input matrix.

inline constexpr bool is_2d_dist(El::Dist coldist, El::Dist rowdist) noexcept
{
    return (coldist == El::MC || coldist == El::MR || rowdist == El::MC
            || rowdist == El::MR);
}

inline constexpr bool is_1d_dist(El::Dist coldist, El::Dist rowdist) noexcept
{
    return (coldist == El::VC || coldist == El::VR || rowdist == El::VC
            || rowdist == El::VR);
}

inline constexpr Distribution to_h2_dist(El::Dist d) noexcept
{
    switch (d)
    {
    case El::MC:
    case El::MR:
    case El::VC:
    case El::VR: return Distribution::Block;
    case El::STAR: return Distribution::Replicated;
    case El::CIRC: return Distribution::Single;
    default: return Distribution::Undefined;
    }
}

inline El::mpi::Comm const&
logical_1d_comm(El::Grid const& g, El::Dist coldist, El::Dist rowdist) noexcept
{
    if ((coldist == El::MR && rowdist == El::MC)
        || (coldist == El::MR && rowdist == El::STAR)
        || (coldist == El::STAR && rowdist == El::MC)
        || (coldist == El::STAR && rowdist == El::VR)
        || (coldist == El::VR && rowdist == El::STAR))
    {
        return g.VRComm();
    }
    return g.VCComm();
}

inline ShapeTuple
make_canonical_grid_shape(El::Grid const& g, El::Dist coldist, El::Dist rowdist)
{
#define MATCHDIST_RETURN_VAL(CDIST, RDIST, ...)                                \
    do                                                                         \
    {                                                                          \
        if (coldist == CDIST && rowdist == RDIST)                              \
        {                                                                      \
            return __VA_ARGS__;                                                \
        }                                                                      \
    } while (0)

    using namespace El::DistNS;
    MATCHDIST_RETURN_VAL(MC, MR, {g.MCSize(), g.MRSize()});
    MATCHDIST_RETURN_VAL(MC, STAR, {g.MCSize(), g.MRSize()});
    MATCHDIST_RETURN_VAL(MR, MC, {g.MRSize(), g.MCSize()});
    MATCHDIST_RETURN_VAL(MR, STAR, {g.MRSize(), g.MCSize()});
    MATCHDIST_RETURN_VAL(STAR, MC, {g.MRSize(), g.MCSize()});
    MATCHDIST_RETURN_VAL(STAR, MR, {g.MCSize(), g.MRSize()});
    MATCHDIST_RETURN_VAL(STAR, VC, {1, g.VCSize()});
    MATCHDIST_RETURN_VAL(STAR, VR, {1, g.VRSize()});
    MATCHDIST_RETURN_VAL(VC, STAR, {g.VCSize(), 1});
    MATCHDIST_RETURN_VAL(VR, STAR, {g.VRSize(), 1});
    MATCHDIST_RETURN_VAL(STAR, STAR, {g.VCSize(), 1});
    MATCHDIST_RETURN_VAL(CIRC, CIRC, {g.VCSize(), 1});
#undef MATCHDIST_RETURN_VAL
    throw std::logic_error("Unknown Hydrogen distribution.");
}

inline ProcessorGrid
make_default_grid(El::Grid const& g, El::Dist coldist, El::Dist rowdist)
{
    auto const& comm = logical_1d_comm(g, coldist, rowdist);
    return ProcessorGrid{comm, make_canonical_grid_shape(g, coldist, rowdist)};
}

template <El::Device D, typename T, typename U>
auto make_dist_tensor_impl(T* const buffer,
                           El::AbstractDistMatrix<U> const& mat)
{
    static_assert(meta::Eq<std::decay_t<T>, U>);
    auto const coldist = mat.ColDist();
    auto const rowdist = mat.RowDist();
    return DistTensor<U>{
        H2Device<D>,
        buffer,
        ShapeTuple{safe_as<int>(mat.Height()), safe_as<int>(mat.Width())},
        DimensionTypeTuple{DimensionType::Any, DimensionType::Any},
        make_default_grid(mat.Grid(), coldist, rowdist),
        DistributionTypeTuple{to_h2_dist(coldist), to_h2_dist(rowdist)},
        ShapeTuple{safe_as<int>(mat.LocalHeight()),
                   safe_as<int>(mat.LocalWidth())},
        StrideTuple{1, mat.LDim()},
        ComputeStream{El::SyncInfoFromMatrix(
            static_cast<El::Matrix<U, D> const&>(mat.LockedMatrix()))}};
}

template <El::Device D, typename T, typename U>
auto make_dist_tensor_ptr_impl(T* const buffer,
                               El::AbstractDistMatrix<U> const& mat)
{
    return std::make_unique<DistTensor<U>>(
        make_dist_tensor_impl<D>(buffer, mat));
}
} // namespace internal

/** @brief Zero-copy convert a Hydrogen distributed matrix to a
 *         DiHydrogen distributed tensor.
 *
 *  A suitable `ProcessorGrid` is constructed on-the-fly.
 *  This costs a collective `MPI_Comm_dup` over a communicator
 *  that is `MPI_SIMILAR` to `mat.Grid().Comm()` when the
 *  `ProcessorGrid` is constructed.
 *
 *  The compute stream is inherited from the SyncInfo object
 *  associated with the Hydrogen matrix.
 *
 *  The returned tensor is a (mutable) view into the memory held by
 *  the input matrix. The distribution is structured such that it is
 *  logically consistent with the underlying distribution of the input
 *  matrix. It will always be 2D even if the underlying distribution
 *  is inherently 1D (e.g, (VC, STAR) and company).
 */
template <typename T>
auto as_h2_tensor(El::AbstractDistMatrix<T>& mat) -> DistTensor<T>
{
    switch (mat.GetLocalDevice())
    {
    case El::Device::CPU:
        return internal::make_dist_tensor_impl<El::Device::CPU>(mat.Buffer(),
                                                                mat);
#ifdef H2_HAS_GPU
    case El::Device::GPU:
        return internal::make_dist_tensor_impl<El::Device::GPU>(mat.Buffer(),
                                                                mat);
#endif
    default: throw std::logic_error("Unknown device.");
    }
}

/** @brief Zero-copy convert a Hydrogen distributed matrix to a
 *         DiHydrogen distributed tensor.
 *
 *  A suitable `ProcessorGrid` is constructed on-the-fly.
 *  This costs a collective `MPI_Comm_dup` over a communicator
 *  that is `MPI_SIMILAR` to `mat.Grid().Comm()` when the
 *  `ProcessorGrid` is constructed.
 *
 *  The compute stream is inherited from the SyncInfo object
 *  associated with the Hydrogen matrix.
 *
 *  The returned tensor is a constant view into the memory held by
 *  the input matrix. The distribution is structured such that it is
 *  logically consistent with the underlying distribution of the input
 *  matrix. It will always be 2D even if the underlying distribution
 *  is inherently 1D (e.g, (VC, STAR) and company).
 */
template <typename T>
auto as_h2_tensor(El::AbstractDistMatrix<T> const& mat) -> DistTensor<T>
{
    switch (mat.GetLocalDevice())
    {
    case El::Device::CPU:
        return internal::make_dist_tensor_impl<El::Device::CPU>(
            mat.LockedBuffer(), mat);
#ifdef H2_HAS_GPU
    case El::Device::GPU:
        return internal::make_dist_tensor_impl<El::Device::GPU>(
            mat.LockedBuffer(), mat);
#endif
    default: throw std::logic_error("Unknown device.");
    }
}

/** @brief Zero-copy convert a Hydrogen distributed matrix to a
 *         DiHydrogen distributed tensor. (Pointer version)
 **/
template <typename T>
auto as_h2_tensor_ptr(El::AbstractDistMatrix<T>& mat)
    -> std::unique_ptr<DistTensor<T>>
{
    switch (mat.GetLocalDevice())
    {
    case El::Device::CPU:
        return internal::make_dist_tensor_ptr_impl<El::Device::CPU>(
            mat.Buffer(), mat);
#ifdef H2_HAS_GPU
    case El::Device::GPU:
        return internal::make_dist_tensor_ptr_impl<El::Device::GPU>(
            mat.Buffer(), mat);
#endif
    default: throw std::logic_error("Unknown device.");
    }
}

/** @brief Zero-copy convert a Hydrogen distributed matrix to a
 *         DiHydrogen distributed tensor. (Const pointer version)
 **/
template <typename T>
auto as_h2_tensor_ptr(El::AbstractDistMatrix<T> const& mat)
    -> std::unique_ptr<DistTensor<T>>
{
    switch (mat.GetLocalDevice())
    {
    case El::Device::CPU:
        return internal::make_dist_tensor_ptr_impl<El::Device::CPU>(
            mat.LockedBuffer(), mat);
#ifdef H2_HAS_GPU
    case El::Device::GPU:
        return internal::make_dist_tensor_ptr_impl<El::Device::GPU>(
            mat.LockedBuffer(), mat);
#endif
    default: throw std::logic_error("Unknown device.");
    }
}

// Now we go the other way. In this case, we require that the user
// provide us an `El::Grid` to use. The reason for this is lifetime
// management of the `El::Grid`. Specifically, `El::DistMatrix`
// doesn't manage the lifetime of the grid at all, so without a
// user-provided option, we'd have to return the pair {Matrix, Grid}.
// We could do that if needed, but for now, let's not.
//
// We also require that the user specify the *exact* distribution they
// want. The reason for this is to resolve ambiguities present in the
// weaker distribution semantics of H2. Suppose you have a matrix that
// is (Block, Replicated) distributed over a (N, 1) grid -- is this a
// (VC, STAR) or a (VR, STAR)? It could also be (MC, STAR) where
// grid.MRSize()==1. Without user input, there's no way to
// disambiguate these cases.

namespace internal
{

template <typename T>
void assert_valid_conversion_to_h(DistTensor<T> const& tensor,
                                  El::Grid const& grid,
                                  El::Dist coldist,
                                  El::Dist rowdist)
{
    // General tensor things

    // This could be relaxed to 1-or-2D and we could pad with a "1".
    // But which dim? From a certain point of view, there is a best
    // answer, but we cannot know the user's intent and either choice
    // is valid. So we defer to the user to straighten this out.
    H2_ASSERT(tensor.ndim() == 2,
              std::logic_error,
              "Tensor must be 2D to zero-copy-convert to Hydrogen");

    H2_ASSERT(
        (tensor.distribution()
         == DistributionTypeTuple{to_h2_dist(coldist), to_h2_dist(rowdist)}),
        std::logic_error,
        "Tensor distribution must be compatible with desired Hydrogen "
        "distribution.");

    if (!tensor.is_local_empty())
    {
        H2_ASSERT(tensor.const_local_tensor().stride(0) == 1,
                  std::logic_error,
                  "Tensor must be fully-packed in the fastest index.");
    }

    // Grid things
    H2_ASSERT(tensor.proc_grid().comm().Size() == grid.Size(),
              std::logic_error,
              "Grid and process grid must be same size.");

    // The comms need to be congruent, not just similar, so that the
    // h2-blocked <-> round-robin local indices are sufficiently
    // aligned. This does not matter for (o,o) or (*,*).
    if (is_1d_dist(coldist, rowdist) || is_2d_dist(coldist, rowdist))
    {
        H2_ASSERT(
            (El::mpi::Congruent(tensor.proc_grid().comm(),
                                logical_1d_comm(grid, coldist, rowdist))),
            std::logic_error,
            "Process grid comm not compatible with requested distribution.");
        H2_ASSERT((tensor.proc_grid().shape()
                   == make_canonical_grid_shape(grid, coldist, rowdist)),
                  std::logic_error,
                  "Grid and process grid must be same shape.");
    }
}

// Return as a ElementalMatrix<T> so we get the Attach/LockedAttach
// member functions.
template <typename T, El::Device D>
auto make_distmat(El::Grid const& g,
                  El::Dist coldist,
                  El::Dist rowdist,
                  El::Int root = 0) -> std::unique_ptr<El::ElementalMatrix<T>>
{
    using namespace El;
#define CHECK_AND_RETURN_IF(COLDIST, ROWDIST)                                  \
    do                                                                         \
    {                                                                          \
        if (coldist == COLDIST && rowdist == ROWDIST)                          \
        {                                                                      \
            return std::make_unique<                                           \
                DistMatrix<T, COLDIST, ROWDIST, ELEMENT, D>>(g, root);         \
        }                                                                      \
    } while (0)

    CHECK_AND_RETURN_IF(MC, MR);
    CHECK_AND_RETURN_IF(MC, STAR);
    CHECK_AND_RETURN_IF(MR, MC);
    CHECK_AND_RETURN_IF(MR, STAR);
    CHECK_AND_RETURN_IF(STAR, MC);
    CHECK_AND_RETURN_IF(STAR, MR);
    CHECK_AND_RETURN_IF(STAR, VC);
    CHECK_AND_RETURN_IF(STAR, VR);
    CHECK_AND_RETURN_IF(VC, STAR);
    CHECK_AND_RETURN_IF(VR, STAR);
    CHECK_AND_RETURN_IF(CIRC, CIRC);
    CHECK_AND_RETURN_IF(STAR, STAR);
#undef CHECK_AND_RETURN_IF
    throw std::logic_error("Unsupported distribution requested.");
}

// Handles device dispatch layer in creating a new matrix object.
template <typename T>
auto make_distmat(Device D,
                  El::Grid const& g,
                  El::Dist coldist,
                  El::Dist rowdist,
                  El::Int root = 0)
{
    switch (D)
    {
    case Device::CPU:
        return make_distmat<T, El::Device::CPU>(g, coldist, rowdist, root);
#ifdef H2_HAS_GPU
    case Device::GPU:
        return make_distmat<T, El::Device::GPU>(g, coldist, rowdist, root);
#endif
    default: throw std::logic_error("Unknown device type.");
    }
}

// Handles attaching the actually memory buffer to the matrix.
template <typename T, typename U>
void as_h_matrix_impl(T* buffer,
                      El::ElementalMatrix<U>& mat,
                      El::Int height,
                      El::Int width,
                      El::Int ldim)
{
    static_assert(meta::Eq<std::decay_t<T>, U>);

    static constexpr El::Int col_align = 0;
    static constexpr El::Int row_align = 0;
    if constexpr (std::is_const_v<T>)
        mat.LockedAttach(
            height, width, mat.Grid(), col_align, row_align, buffer, ldim);
    else
        mat.Attach(
            height, width, mat.Grid(), col_align, row_align, buffer, ldim);
}

// Sets up the synchronization mechanics for the matrix.
template <typename T>
void set_sync(El::AbstractDistMatrix<T>& mat, ComputeStream stream)
{
    // Hydrogen doesn't have "real" sync objects for CPU, so we only
    // care about the GPU case.
#ifdef H2_HAS_GPU
    if (mat.GetLocalDevice() == El::Device::GPU)
    {
        constexpr auto D = El::Device::GPU;
        auto& loc = static_cast<El::Matrix<T, D>&>(mat.Matrix());
        El::SetSyncInfo(loc,
                        El::SyncInfo<D>{stream.get_stream<D>(),
                                        internal::get_default_event<D>()});
    }
#endif
}
} // namespace internal

/** @brief Zero-copy convert a distributed tensor into a Hydrogen
 *         distributed matrix.
 *
 *  This function allows one to effectively reinterpret certain
 *  distributed H2 tensors as Hydrogen distributed matrices, provided
 *  certain criteria are met. The criteria are intentionally defined
 *  very narrowly to limit the potential for abuse and to maximize the
 *  likelihood of consistency-by-default.
 *
 *  The first requirement is that only 2D tensors are reinterpretable
 *  this way. One might argue that we could accept 1D tensors and just
 *  pad with a fake dimension of size 1 - but which dimension? (There
 *  is an obvious way to answer this, but nonetheless...) We answer
 *  this by deferring to the user, since reshaping an H2 tensor is
 *  trivial.
 *
 *  The next requirement is that the requested Elemental distribution
 *  must be compatible with the H2 distribution of the tensor. This is
 *  to ensure consistency with zero copy.
 *
 *  To further ensure consistency of the parallel distributions, the
 *  respective communicator objects must be congruent. Currently this
 *  is implemented very aggressively: the underlying MPI communicator
 *  of the `h2::DistTensor`'s `ProcessorGrid` must be `MPI_CONGRUENT`
 *  to the input grid's `VCComm` or `VRComm`, depending on if the
 *  distribution is column-major or row-major.
 *
 *  Finally, local Elemental matrices must be fully-packed in their
 *  fastest-moving index. That is, they must have packed columns.
 *  Thus, the fastest-moving index of the local tensor must be fully
 *  packed to ensure zero-copy conversion is sensible.
 *
 *  @param[in] tensor The tensor to convert to Hydrogen format
 *  @param[in] g The grid over which the distribution is posed
 *  @param[in] coldist The distribution strategy for a column
 *  @param[in] rowdist the distribution strategy for a row
 *
 *  @returns A pointer to a new Hydrogen AbstractDistMatrix that is a
 *           view of the input tensor's data.
 */
template <typename T>
auto as_h_matrix(DistTensor<T>& tensor,
                 El::Grid const& g,
                 El::Dist coldist,
                 El::Dist rowdist) -> std::unique_ptr<El::AbstractDistMatrix<T>>
{
    using namespace internal;
    assert_valid_conversion_to_h(tensor, g, coldist, rowdist);
    // Make the new DistMatrix
    auto out = make_distmat<T>(tensor.get_device(), g, coldist, rowdist);
    // Attach data
    as_h_matrix_impl(tensor.data(),
                     *out,
                     tensor.shape(0),
                     tensor.shape(1),
                     tensor.const_local_tensor().stride(1));
    // Update syncinfo
    set_sync(*out, tensor.get_stream());
    return out;
}

/** @copydoc as_h_matrix()
 */
template <typename T>
auto as_h_matrix(DistTensor<T> const& tensor,
                 El::Grid const& g,
                 El::Dist coldist,
                 El::Dist rowdist) -> std::unique_ptr<El::AbstractDistMatrix<T>>
{
    using namespace internal;
    assert_valid_conversion_to_h(tensor, g, coldist, rowdist);
    // Make the new DistMatrix
    auto out = make_distmat<T>(tensor.get_device(), g, coldist, rowdist);
    // Attach data
    as_h_matrix_impl(tensor.data(),
                     *out,
                     tensor.shape(0),
                     tensor.shape(1),
                     tensor.const_local_tensor().stride(1));
    // Update syncinfo
    set_sync(*out, tensor.get_stream());
    return out;
}
} // namespace h2
