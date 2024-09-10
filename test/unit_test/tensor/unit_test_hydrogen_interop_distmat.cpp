#include "h2/tensor/dist_tensor.hpp"
#include "h2/tensor/hydrogen_interop.hpp"
#include "h2/tensor/proc_grid.hpp"
#include "utils.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

// Simplify syntax
using Catch::Matchers::ContainsSubstring;

// For each case, I must show that the following scenarios play out correctly:
//
//   1. Valid tensor/matrix, ad hoc proc grid
//   2. Valid tensor/matrix, valid proc grid
//   3. Valid tensor/matrix, invalid proc grid [THROWS]
//   4. Invalid tensor/matrix [THROWS], any proc grid (not checked)

template <typename T, typename DT>
struct MatDev
{
  using type = T;
  static constexpr auto device = DT::value;
};

template <typename DT>
using AllMats =
  h2::meta::TL<MatDev<El::AbstractDistMatrix<DataType>, DT>,
               MatDev<El::AbstractDistMatrix<DataType> const, DT>>;

using AllMatDevs =
  h2::meta::tlist::Flatten<h2::meta::tlist::MapTL<AllMats, AllDevList>>;

TEMPLATE_LIST_TEST_CASE(
  "Hydrogen DistMatrix to DiHydrogen DistTensor conversion",
  "[dist-tensor][h_h2]",
  AllMatDevs)
{
  using AbsDistMatT = typename TestType::type;
  constexpr h2::Device Dev = TestType::device;
  constexpr hydrogen::Device HDev = h2::HydrogenDevice<Dev>;

  El::Grid grid;  // COMM_WORLD, as square as possible, column-major ordering
  SECTION("DistMatrix(STAR, STAR)")
  {
    El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, HDev> A(grid);
    El::Uniform(A, grid.Height() * 4, grid.Width() * 5);

    AbsDistMatT& A_ref = A;
    auto A_tensor = h2::as_h2_tensor_ptr(A_ref);

    // These are "the usual" things we expect regardless of
    // distribution. These assertions verify that the "attaching"
    // constructor worked as expected.
    REQUIRE((bool) A_tensor);
    CHECK(A_tensor->ndim() == 2UL);
    CHECK(A_tensor->numel() == A.Height() * A.Width());
    CHECK(A_tensor->shape(0) == A.Height());
    CHECK(A_tensor->shape(1) == A.Width());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    // This is specific to STAR,STAR
    CHECK(A_tensor->local_shape(0) == A_tensor->shape(0));
    CHECK(A_tensor->local_shape(1) == A_tensor->shape(1));
    CHECK(A_tensor->distribution(0) == h2::Distribution::Replicated);
    CHECK(A_tensor->distribution(1) == h2::Distribution::Replicated);
  }

  SECTION("DistMatrix(STAR, VC)")
  {
    El::DistMatrix<DataType, El::STAR, El::VC, El::ELEMENT, HDev> A(grid);
    El::Uniform(A, grid.Height() * 4, grid.Width() * 5);

    AbsDistMatT& A_ref = A;
    auto A_tensor = h2::as_h2_tensor_ptr(A_ref);

    REQUIRE((bool) A_tensor);
    CHECK(A_tensor->ndim() == 2UL);
    CHECK(A_tensor->numel() == A.Height() * A.Width());
    CHECK(A_tensor->shape(0) == A.Height());
    CHECK(A_tensor->shape(1) == A.Width());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    // The row index is "replicated" on each process while the
    // column index is "block"-distributed
    CHECK(A_tensor->local_shape(0) == A_tensor->shape(0));
    CHECK(A_tensor->local_shape(0) == A.Height());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    CHECK(A_tensor->distribution(0) == h2::Distribution::Replicated);
    CHECK(A_tensor->distribution(1) == h2::Distribution::Block);
  }

  SECTION("DistMatrix(STAR, VR)")
  {
    El::DistMatrix<DataType, El::STAR, El::VR, El::ELEMENT, HDev> A(grid);
    El::Uniform(A, grid.Height() * 4, grid.Width() * 5);

    AbsDistMatT& A_ref = A;
    auto A_tensor = h2::as_h2_tensor_ptr(A_ref);
    REQUIRE((bool) A_tensor);
    CHECK(A_tensor->ndim() == 2UL);
    CHECK(A_tensor->numel() == A.Height() * A.Width());
    CHECK(A_tensor->shape(0) == A.Height());
    CHECK(A_tensor->shape(1) == A.Width());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    // The row index is "replicated" on each process while the
    // column index is "block"-distributed
    CHECK(A_tensor->local_shape(0) == A_tensor->shape(0));
    CHECK(A_tensor->local_shape(0) == A.Height());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    CHECK(A_tensor->distribution(0) == h2::Distribution::Replicated);
    CHECK(A_tensor->distribution(1) == h2::Distribution::Block);
  }

  SECTION("DistMatrix(VC, STAR)")
  {
    El::DistMatrix<DataType, El::VC, El::STAR, El::ELEMENT, HDev> A(grid);
    El::Uniform(A, grid.Height() * 4, grid.Width() * 5);

    AbsDistMatT& A_ref = A;
    auto A_tensor = h2::as_h2_tensor_ptr(A_ref);
    REQUIRE((bool) A_tensor);
    CHECK(A_tensor->ndim() == 2UL);
    CHECK(A_tensor->numel() == A.Height() * A.Width());
    CHECK(A_tensor->shape(0) == A.Height());
    CHECK(A_tensor->shape(1) == A.Width());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    // The column index is "replicated" on each process while the
    // row index is "block"-distributed

    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());
    CHECK(A_tensor->local_shape(1) == A.Width());
    CHECK(A_tensor->local_shape(1) == A_tensor->shape(1));

    CHECK(A_tensor->distribution(0) == h2::Distribution::Block);
    CHECK(A_tensor->distribution(1) == h2::Distribution::Replicated);
  }

  SECTION("DistMatrix(VR, STAR)")
  {
    El::DistMatrix<DataType, El::VR, El::STAR, El::ELEMENT, HDev> A(grid);
    El::Uniform(A, grid.Height() * 4, grid.Width() * 5);

    AbsDistMatT& A_ref = A;
    auto A_tensor = h2::as_h2_tensor_ptr(A_ref);
    REQUIRE((bool) A_tensor);
    CHECK(A_tensor->ndim() == 2UL);
    CHECK(A_tensor->numel() == A.Height() * A.Width());
    CHECK(A_tensor->shape(0) == A.Height());
    CHECK(A_tensor->shape(1) == A.Width());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    // The column index is "replicated" on each process while the
    // row index is "block"-distributed.
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());
    CHECK(A_tensor->local_shape(1) == A.Width());
    CHECK(A_tensor->local_shape(1) == A_tensor->shape(1));
    CHECK(A_tensor->distribution(0) == h2::Distribution::Block);
    CHECK(A_tensor->distribution(1) == h2::Distribution::Replicated);
  }

  // // Now the 2D distributions

  SECTION("DistMatrix(STAR, MC)")
  {
    El::DistMatrix<DataType, El::STAR, El::MC, El::ELEMENT, HDev> A(grid);
    El::Uniform(A, grid.Height() * 4, grid.Width() * 5);

    AbsDistMatT& A_ref = A;
    auto A_tensor = h2::as_h2_tensor_ptr(A_ref);
    REQUIRE((bool) A_tensor);
    CHECK(A_tensor->ndim() == 2UL);
    CHECK(A_tensor->numel() == A.Height() * A.Width());
    CHECK(A_tensor->shape(0) == A.Height());
    CHECK(A_tensor->shape(1) == A.Width());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    // The row index is "replicated" on each process while the
    // column index is "block" distributed down a column of the
    // processing grid. (Each row of the processing grid contains
    // an identical slice of the matrix.)

    CHECK(A_tensor->distribution(0) == h2::Distribution::Replicated);
    CHECK(A_tensor->distribution(1) == h2::Distribution::Block);

    CHECK(A_tensor->local_shape(0) == A.Height());
  }

  SECTION("DistMatrix(STAR, MR)")
  {
    El::DistMatrix<DataType, El::STAR, El::MR, El::ELEMENT, HDev> A(grid);
    El::Uniform(A, grid.Height() * 4, grid.Width() * 5);

    AbsDistMatT& A_ref = A;
    auto A_tensor = h2::as_h2_tensor_ptr(A_ref);
    REQUIRE((bool) A_tensor);
    CHECK(A_tensor->ndim() == 2UL);
    CHECK(A_tensor->numel() == A.Height() * A.Width());
    CHECK(A_tensor->shape(0) == A.Height());
    CHECK(A_tensor->shape(1) == A.Width());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    // The row index is "replicated" on each process while the
    // column index is "block" distributed across a row of the
    // processing grid. (Each column of the processing grid
    // contains an identical slice of the matrix.)

    CHECK(A_tensor->distribution(0) == h2::Distribution::Replicated);
    CHECK(A_tensor->distribution(1) == h2::Distribution::Block);
  }

  SECTION("DistMatrix(MC, STAR)")
  {
    El::DistMatrix<DataType, El::MC, El::STAR, El::ELEMENT, HDev> A(grid);
    El::Uniform(A, grid.Height() * 4, grid.Width() * 5);

    AbsDistMatT& A_ref = A;
    auto A_tensor = h2::as_h2_tensor_ptr(A_ref);
    REQUIRE((bool) A_tensor);
    CHECK(A_tensor->ndim() == 2UL);
    CHECK(A_tensor->numel() == A.Height() * A.Width());
    CHECK(A_tensor->shape(0) == A.Height());
    CHECK(A_tensor->shape(1) == A.Width());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    // The row index is block-distributed down a column of the
    // processing grid while the column index is replicated on
    // each process. (Each row of the processing grid contains an
    // identical slice of the matrix.)

    CHECK(A_tensor->local_shape(0)
          == h2::internal::get_dim_local_size<h2::Distribution::Block>(
            A_tensor->shape(0),
            A_tensor->proc_grid().shape(0),
            A_tensor->proc_grid().get_dimension_rank(0),
            false));

    CHECK(A_tensor->local_shape(1) == A_tensor->shape(1));
    CHECK(A_tensor->local_shape(1) == A.Width());

    CHECK(A_tensor->distribution(0) == h2::Distribution::Block);
    CHECK(A_tensor->distribution(1) == h2::Distribution::Replicated);
  }

  SECTION("DistMatrix(MR, STAR)")
  {
    El::DistMatrix<DataType, El::MR, El::STAR, El::ELEMENT, HDev> A(grid);
    El::Uniform(A, grid.Height() * 4, grid.Width() * 5);

    AbsDistMatT& A_ref = A;
    auto A_tensor = h2::as_h2_tensor_ptr(A_ref);
    REQUIRE((bool) A_tensor);
    CHECK(A_tensor->ndim() == 2UL);
    CHECK(A_tensor->numel() == A.Height() * A.Width());
    CHECK(A_tensor->shape(0) == A.Height());
    CHECK(A_tensor->shape(1) == A.Width());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    // The row index is block-distribued across a row of the
    // processing grid while the column index is replicated on
    // each process. (Each column of the processing grid contains
    // an identical slice of the matrix.)

    CHECK(A_tensor->local_shape(1) == A_tensor->shape(1));
    CHECK(A_tensor->local_shape(1) == A.Width());

    CHECK(A_tensor->distribution(0) == h2::Distribution::Block);
    CHECK(A_tensor->distribution(1) == h2::Distribution::Replicated);
  }

  SECTION("DistMatrix(MC, MR)")
  {
    El::DistMatrix<DataType, El::MC, El::MR, El::ELEMENT, HDev> A(grid);
    El::Uniform(A, grid.Height() * 4, grid.Width() * 5);

    AbsDistMatT& A_ref = A;
    auto A_tensor = h2::as_h2_tensor_ptr(A_ref);
    REQUIRE((bool) A_tensor);
    CHECK(A_tensor->ndim() == 2UL);
    CHECK(A_tensor->numel() == A.Height() * A.Width());
    CHECK(A_tensor->shape(0) == A.Height());
    CHECK(A_tensor->shape(1) == A.Width());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    // The row index is block-distributed down a column of the
    // processing grid while the column index is block-distributed
    // across a row of the processing grid.

    CHECK(A_tensor->distribution(0) == h2::Distribution::Block);
    CHECK(A_tensor->distribution(1) == h2::Distribution::Block);
  }

  SECTION("DistMatrix(MR, MC)")
  {
    El::DistMatrix<DataType, El::MR, El::MC, El::ELEMENT, HDev> A(grid);
    El::Uniform(A, grid.Height() * 4, grid.Width() * 5);

    AbsDistMatT& A_ref = A;
    auto A_tensor = h2::as_h2_tensor_ptr(A_ref);
    REQUIRE((bool) A_tensor);
    CHECK(A_tensor->ndim() == 2UL);
    CHECK(A_tensor->numel() == A.Height() * A.Width());
    CHECK(A_tensor->shape(0) == A.Height());
    CHECK(A_tensor->shape(1) == A.Width());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    // The row index is block-distributed across a row of the
    // processing grid while the column index is block-distributed
    // down a column of the processing grid.

    CHECK(A_tensor->distribution(0) == h2::Distribution::Block);
    CHECK(A_tensor->distribution(1) == h2::Distribution::Block);
  }

  SECTION("DistMatrix(CIRC, CIRC)")
  {
    El::DistMatrix<DataType, El::CIRC, El::CIRC, El::ELEMENT, HDev> A(grid);
    El::Uniform(A, grid.Height() * 4, grid.Width() * 5);

    AbsDistMatT& A_ref = A;
    auto A_tensor = h2::as_h2_tensor_ptr(A_ref);
    REQUIRE((bool) A_tensor);
    CHECK(A_tensor->ndim() == 2UL);
    CHECK(A_tensor->numel() == A.Height() * A.Width());
    CHECK(A_tensor->shape(0) == A.Height());
    CHECK(A_tensor->shape(1) == A.Width());
    CHECK(A_tensor->local_shape(0) == A.LocalHeight());
    CHECK(A_tensor->local_shape(1) == A.LocalWidth());

    // The matrix is gathered to the root process of the grid.
    if (grid.Rank() == 0)
    {
      CHECK(A_tensor->local_shape(0) == A_tensor->shape(0));
      CHECK(A_tensor->local_shape(1) == A_tensor->shape(1));
    }
    else
    {
      CHECK(A_tensor->local_shape(0) == 0);
      CHECK(A_tensor->local_shape(1) == 0);
    }

    CHECK(A_tensor->distribution(0) == h2::Distribution::Single);
    CHECK(A_tensor->distribution(1) == h2::Distribution::Single);
  }

  // These are not supported. They're weird.
  SECTION("DistMatrix(MD, STAR) -- NOT supported")
  {
    El::DistMatrix<DataType, El::MD, El::STAR, El::ELEMENT, HDev> A(grid);
    AbsDistMatT& A_ref = A;
    CHECK_THROWS_WITH(h2::as_h2_tensor(A_ref),
                      ContainsSubstring("Unknown Hydrogen distribution"));
  }
  SECTION("DistMatrix(STAR, MD) -- NOT supported")
  {
    El::DistMatrix<DataType, El::STAR, El::MD, El::ELEMENT, HDev> A(grid);
    AbsDistMatT& A_ref = A;
    CHECK_THROWS_WITH(h2::as_h2_tensor(A_ref),
                      ContainsSubstring("Unknown Hydrogen distribution"));
  }
}

// H2 -> H

template <typename T, typename DT>
using TensorDev = MatDev<T, DT>;

template <typename DT>
using AllTensors = h2::meta::TL<TensorDev<h2::DistTensor<DataType>, DT>,
                                TensorDev<h2::DistTensor<DataType> const, DT>>;

using AllTensorDevs =
  h2::meta::tlist::Flatten<h2::meta::tlist::MapTL<AllTensors, AllDevList>>;

// Should test:
//
//  1. Cases that should work do.
//  2. Bad process grid shapes fail (when possible)
//  3. Dist types on tensor that don't match requested El dist fail
//  4. non-2d tensors fail
//
TEMPLATE_LIST_TEST_CASE(
  "Hydrogen DistMatrix to DiHydrogen DistTensor conversion",
  "[dist-tensor][h2_h]",
  AllTensorDevs)
{
  using TensorType = typename TestType::type;
  constexpr auto device = TestType::device;

  /* BEGIN This is consistent across all tests */

  constexpr auto who_cares = h2::DimensionType::Any;
  constexpr auto dim_types = h2::DimensionTypeTuple{who_cares, who_cares};

  // Let Elemental shape the grid. It will be as square as possible.
  El::Grid const g{El::mpi::NewWorldComm()};

  // The relevant grid shapes:
  h2::ShapeTuple const grid_shape_cmaj = {g.Height(), g.Width()};
  h2::ShapeTuple const grid_shape_rmaj = {g.Width(), g.Height()};
  h2::ShapeTuple const grid_shape_1dc = {g.Size(), 1};
  h2::ShapeTuple const grid_shape_1dr = {1, g.Size()};

  h2::ShapeTuple const tensor_shape{
    3 * grid_shape_cmaj[0] + grid_shape_cmaj[0] / 2,
    5 * grid_shape_cmaj[1] + grid_shape_cmaj[1] / 2};
  /* END consistent data */

  SECTION("To DistMatrix(MC, MR)")
  {
    constexpr auto coldist = El::MC;
    constexpr auto rowdist = El::MR;

    constexpr auto dist_types = h2::DistributionTypeTuple{
      h2::internal::to_h2_dist(coldist), h2::internal::to_h2_dist(rowdist)};

    SECTION("Correct communicator and dist type")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_cmaj};

      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    // Test that bad grid shapes fail
    SECTION("Correct dist types, communicator is wrong shape")
    {
      if (grid_shape_rmaj == grid_shape_cmaj)
        SKIP("Row major and column major communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_rmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d col comm instead of 2d")
    {
      if (grid_shape_cmaj == grid_shape_1dc)
        SKIP("2d and 1d column communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_1dc};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d row comm instead of 2d")
    {
      if (grid_shape_cmaj == grid_shape_1dr)
        SKIP("2d and 1d row communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_1dr};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, incongruent communicator")
    {
      if (g.Height() == 1 || g.Width() == 1)
        SKIP("1-D communicator");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_cmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }
  }

  SECTION("To DistMatrix(MC, STAR)")
  {
    constexpr auto coldist = El::MC;
    constexpr auto rowdist = El::STAR;

    constexpr auto dist_types = h2::DistributionTypeTuple{
      h2::internal::to_h2_dist(coldist), h2::internal::to_h2_dist(rowdist)};

    SECTION("Correct communicator and dist type")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_cmaj};

      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    // Test that bad grid shapes fail
    SECTION("Correct dist types, communicator is wrong shape")
    {
      if (grid_shape_rmaj == grid_shape_cmaj)
        SKIP("Row major and column major communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_rmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d col comm instead of 2d")
    {
      if (grid_shape_cmaj == grid_shape_1dc)
        SKIP("2d and 1d column communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_1dc};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d row comm instead of 2d")
    {
      if (grid_shape_cmaj == grid_shape_1dr)
        SKIP("2d and 1d row communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_1dr};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, incongruent communicator")
    {
      if (g.Height() == 1 || g.Width() == 1)
        SKIP("1-D communicator");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_cmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }
  }

  SECTION("To DistMatrix(MR, MC)")
  {
    constexpr auto coldist = El::MR;
    constexpr auto rowdist = El::MC;

    constexpr auto dist_types = h2::DistributionTypeTuple{
      h2::internal::to_h2_dist(coldist), h2::internal::to_h2_dist(rowdist)};

    SECTION("Correct communicator and dist type")
    {
      // This is a 2D row-major distribution.
      auto const proc_grid = h2::ProcessorGrid{g.VRComm(), grid_shape_rmaj};

      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    SECTION("Correct dist types, communicator is wrong shape")
    {
      if (grid_shape_rmaj == grid_shape_cmaj)
        SKIP("Row major and column major communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_cmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d col comm instead of 2d")
    {
      if (grid_shape_rmaj == grid_shape_1dc)
        SKIP("2d and 1d column communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_1dc};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d row comm instead of 2d")
    {
      if (grid_shape_rmaj == grid_shape_1dr)
        SKIP("2d and 1d row communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_1dr};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, incongruent communicator")
    {
      if (g.Height() == 1 || g.Width() == 1)
        SKIP("1-D communicator");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_rmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }
  }

  SECTION("To DistMatrix(MR, STAR)")
  {
    constexpr auto coldist = El::MR;
    constexpr auto rowdist = El::STAR;

    constexpr auto dist_types = h2::DistributionTypeTuple{
      h2::internal::to_h2_dist(coldist), h2::internal::to_h2_dist(rowdist)};

    SECTION("Correct communicator and dist type")
    {
      // This is a 2D row-major distribution.
      auto const proc_grid = h2::ProcessorGrid{g.VRComm(), grid_shape_rmaj};

      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    SECTION("Correct dist types, communicator is wrong shape")
    {
      if (grid_shape_rmaj == grid_shape_cmaj)
        SKIP("Row major and column major communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_cmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d col comm instead of 2d")
    {
      if (grid_shape_rmaj == grid_shape_1dc)
        SKIP("2d and 1d column communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_1dc};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d row comm instead of 2d")
    {
      if (grid_shape_rmaj == grid_shape_1dr)
        SKIP("2d and 1d row communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_1dr};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, incongruent communicator")
    {
      if (g.Height() == 1 || g.Width() == 1)
        SKIP("1-D communicator");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_rmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }
  }

  SECTION("To DistMatrix(STAR, MC)")
  {
    constexpr auto coldist = El::STAR;
    constexpr auto rowdist = El::MC;

    constexpr auto dist_types = h2::DistributionTypeTuple{
      h2::internal::to_h2_dist(coldist), h2::internal::to_h2_dist(rowdist)};

    SECTION("Correct communicator and dist type")
    {
      // This is a 2D row-major distribution.
      auto const proc_grid = h2::ProcessorGrid{g.VRComm(), grid_shape_rmaj};

      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    SECTION("Correct dist types, communicator is wrong shape")
    {
      if (grid_shape_rmaj == grid_shape_cmaj)
        SKIP("Row major and column major communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_cmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d col comm instead of 2d")
    {
      if (grid_shape_rmaj == grid_shape_1dc)
        SKIP("2d and 1d column communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_1dc};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d row comm instead of 2d")
    {
      if (grid_shape_rmaj == grid_shape_1dr)
        SKIP("2d and 1d row communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_1dr};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, incongruent communicator")
    {
      if (g.Height() == 1 || g.Width() == 1)
        SKIP("1-D communicator");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_rmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }
  }

  SECTION("To DistMatrix(STAR, MR)")
  {
    constexpr auto coldist = El::STAR;
    constexpr auto rowdist = El::MR;

    constexpr auto dist_types = h2::DistributionTypeTuple{
      h2::internal::to_h2_dist(coldist), h2::internal::to_h2_dist(rowdist)};

    SECTION("Correct communicator and dist type")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_cmaj};

      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    // Test that bad grid shapes fail
    SECTION("Correct dist types, communicator is wrong shape")
    {
      if (grid_shape_rmaj == grid_shape_cmaj)
        SKIP("Row major and column major communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_rmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d col comm instead of 2d")
    {
      if (grid_shape_cmaj == grid_shape_1dc)
        SKIP("2d and 1d column communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_1dc};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d row comm instead of 2d")
    {
      if (grid_shape_cmaj == grid_shape_1dr)
        SKIP("2d and 1d row communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_1dr};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, incongruent communicator")
    {
      if (g.Height() == 1 || g.Width() == 1)
        SKIP("1-D communicator");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_cmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }
  }

  SECTION("To DistMatrix(STAR, VC)")
  {
    constexpr auto coldist = El::STAR;
    constexpr auto rowdist = El::VC;

    constexpr auto dist_types = h2::DistributionTypeTuple{
      h2::internal::to_h2_dist(coldist), h2::internal::to_h2_dist(rowdist)};

    SECTION("Correct communicator and dist type")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_1dr};

      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    SECTION("Correct dist types, 2d col-major comm instead of 1d row comm")
    {
      if (grid_shape_1dr == grid_shape_cmaj)
        SKIP("1d row comm and column major communicator shapes "
             "coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_cmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 2d row-major comm instead of 1d row comm")
    {
      if (grid_shape_1dr == grid_shape_rmaj)
        SKIP("2d row-major and 1d row communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_rmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d col comm instead of 1d row comm")
    {
      if (grid_shape_1dr == grid_shape_1dc)
        SKIP("1d row and 1d column communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_1dc};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    // This is more complicated. If the 2d comm is 1xN or Nx1,
    // then VCComm and VRComm are actually congruent.
    SECTION("Correct dist types, incongruent communicator")
    {
      if (grid_shape_1dr == grid_shape_rmaj
          || grid_shape_1dr == grid_shape_cmaj)
        SKIP("1d row and col comms are congruent");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_1dr};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }
  }

  SECTION("To DistMatrix(STAR, VR)")
  {
    constexpr auto coldist = El::STAR;
    constexpr auto rowdist = El::VR;

    constexpr auto dist_types = h2::DistributionTypeTuple{
      h2::internal::to_h2_dist(coldist), h2::internal::to_h2_dist(rowdist)};

    // This is a 1D distribution
    // auto const proc_grid = h2::ProcessorGrid{g.VRComm(), grid_shape_1dr};
    //
    // h2::DistTensor<DataType> tensor{
    //     device, tensor_shape, dim_types, proc_grid, dist_types};
    // TensorType& tensor_ref = tensor;
    //
    // auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
    // CHECK((bool) x);

    SECTION("Correct communicator and dist type")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VRComm(), grid_shape_1dr};

      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    SECTION("Correct dist types, 2d col-major comm instead of 1d row comm")
    {
      if (grid_shape_1dr == grid_shape_cmaj)
        SKIP("1d row comm and column major communicator shapes "
             "coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_cmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 2d row-major comm instead of 1d row comm")
    {
      if (grid_shape_1dr == grid_shape_rmaj)
        SKIP("2d row-major and 1d row communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_rmaj};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, 1d col comm instead of 1d row comm")
    {
      if (grid_shape_1dr == grid_shape_1dc)
        SKIP("1d row and 1d column communicator shapes coincide");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VRComm(), grid_shape_1dc};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }

    SECTION("Correct dist types, incongruent communicator")
    {
      if (grid_shape_1dr == grid_shape_rmaj
          || grid_shape_1dr == grid_shape_cmaj)
        SKIP("1d row and col comms are congruent");

      auto const proc_grid_bad = h2::ProcessorGrid{g.VCComm(), grid_shape_1dr};
      h2::DistTensor<DataType> tensor_bad{
        device, tensor_shape, dim_types, proc_grid_bad, dist_types};
      TensorType& tensor_bad_ref = tensor_bad;
      CHECK_THROWS(as_h_matrix(tensor_bad_ref, g, coldist, rowdist));
    }
  }

  SECTION("To DistMatrix(VC, STAR)")
  {
    constexpr auto coldist = El::VC;
    constexpr auto rowdist = El::STAR;

    constexpr auto dist_types = h2::DistributionTypeTuple{
      h2::internal::to_h2_dist(coldist), h2::internal::to_h2_dist(rowdist)};

    // This is a 1D distribution
    auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_1dc};

    h2::DistTensor<DataType> tensor{
      device, tensor_shape, dim_types, proc_grid, dist_types};
    TensorType& tensor_ref = tensor;

    auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
    CHECK((bool) x);
  }

  SECTION("To DistMatrix(VR, STAR)")
  {
    constexpr auto coldist = El::VR;
    constexpr auto rowdist = El::STAR;

    constexpr auto dist_types = h2::DistributionTypeTuple{
      h2::internal::to_h2_dist(coldist), h2::internal::to_h2_dist(rowdist)};

    // This is a 1D distribution
    auto const proc_grid = h2::ProcessorGrid{g.VRComm(), grid_shape_1dc};

    h2::DistTensor<DataType> tensor{
      device, tensor_shape, dim_types, proc_grid, dist_types};
    TensorType& tensor_ref = tensor;

    auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
    CHECK((bool) x);
  }

  SECTION("To DistMatrix(STAR, STAR)")
  {
    constexpr auto coldist = El::STAR;
    constexpr auto rowdist = El::STAR;

    constexpr auto dist_types = h2::DistributionTypeTuple{
      h2::internal::to_h2_dist(coldist), h2::internal::to_h2_dist(rowdist)};

    // Any properly-sized communicator should be fine.
    SECTION("2D column-major communicator")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_cmaj};
      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    SECTION("2D row-major communicator")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_rmaj};
      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    SECTION("1D column communicator")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_1dc};
      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    SECTION("1D row communicator")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_1dr};
      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }
  }

  SECTION("To DistMatrix(CIRC, CIRC)")
  {
    constexpr auto coldist = El::CIRC;
    constexpr auto rowdist = El::CIRC;

    constexpr auto dist_types = h2::DistributionTypeTuple{
      h2::internal::to_h2_dist(coldist), h2::internal::to_h2_dist(rowdist)};

    // Any properly-sized communicator should be fine.
    SECTION("2D column-major communicator")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_cmaj};
      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    SECTION("2D row-major communicator")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_rmaj};
      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    SECTION("1D column communicator")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_1dc};
      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }

    SECTION("1D row communicator")
    {
      auto const proc_grid = h2::ProcessorGrid{g.VCComm(), grid_shape_1dr};
      h2::DistTensor<DataType> tensor{
        device, tensor_shape, dim_types, proc_grid, dist_types};
      TensorType& tensor_ref = tensor;

      auto x = as_h_matrix(tensor_ref, g, coldist, rowdist);
      CHECK((bool) x);
    }
  }
}
