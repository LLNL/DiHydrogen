////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "h2/tensor/tensor.hpp"
#include "h2/tensor/copy.hpp"
#include "utils.hpp"
#include "../mpi_utils.hpp"

using namespace h2;

TEMPLATE_LIST_TEST_CASE("Same-type distributed tensor copy works",
                        "[dist-tensor][dist-copy]",
                        AllDevPairsList)
{
  constexpr Device SrcDev = meta::tlist::At<TestType, 0>::value;
  constexpr Device DstDev = meta::tlist::At<TestType, 1>::value;
  using SrcTensorType = DistTensor<DataType>;
  using DstTensorType = DistTensor<DataType>;
  constexpr DataType src_val = static_cast<DataType>(1);
  constexpr DataType dst_val = static_cast<DataType>(2);

  SECTION("Copying into an existing tensor works without resizing")
  {
    for_comms([&](Comm& comm) {
      for_grid_shapes([&](ShapeTuple grid_shape) {
        for (Distribution dist : {Distribution::Block,
                                  Distribution::Replicated,
                                  Distribution::Single})
        {
          ProcessorGrid grid = ProcessorGrid(comm, grid_shape);
          ShapeTuple tensor_shape(8, 5, 12);
          tensor_shape.set_size(grid.ndim());
          DTTuple tensor_dim_types(TuplePad<DTTuple>(grid.ndim(), DT::Any));
          DistTTuple tensor_dist(TuplePad<DistTTuple>(grid.ndim(), dist));
          SrcTensorType src_tensor = SrcTensorType(
              SrcDev, tensor_shape, tensor_dim_types, grid, tensor_dist);
          DstTensorType dst_tensor = DstTensorType(
              DstDev, tensor_shape, tensor_dim_types, grid, tensor_dist);

          for (DataIndexType i = 0; i < src_tensor.local_numel(); ++i)
          {
            write_ele<SrcDev>(src_tensor.data(), i, src_val);
          }
          for (DataIndexType i = 0; i < dst_tensor.local_numel(); ++i)
          {
            write_ele<DstDev>(dst_tensor.data(), i, dst_val);
          }

          DataType* dst_orig_data = dst_tensor.data();

          REQUIRE_NOTHROW(copy(dst_tensor, src_tensor));

          REQUIRE(dst_tensor.shape() == tensor_shape);
          REQUIRE(dst_tensor.local_shape() == src_tensor.local_shape());
          REQUIRE(dst_tensor.dim_types() == tensor_dim_types);
          REQUIRE(dst_tensor.distribution() == tensor_dist);
          REQUIRE(dst_tensor.numel() == src_tensor.numel());
          REQUIRE_FALSE(dst_tensor.is_empty());
          REQUIRE(dst_tensor.is_local_empty() == src_tensor.is_local_empty());
          REQUIRE_FALSE(dst_tensor.is_view());
          if (src_tensor.is_local_empty())
          {
            REQUIRE(dst_tensor.data() == nullptr);
          }
          else
          {
            REQUIRE(dst_tensor.data() != src_tensor.data());
          }
          REQUIRE(dst_tensor.data() == dst_orig_data);

          for (DataIndexType i = 0; i < src_tensor.local_numel(); ++i)
          {
            REQUIRE(read_ele<SrcDev>(src_tensor.data(), i) == src_val);
            REQUIRE(read_ele<DstDev>(dst_tensor.data(), i) == src_val);
          }
        }
      }, comm);
    });
  }

  SECTION("Copying into a different-sized tensor works")
  {
    for_comms([&](Comm& comm) {
      for_grid_shapes([&](ShapeTuple grid_shape) {
        for (Distribution src_dist : {Distribution::Block,
                                      Distribution::Replicated,
                                      Distribution::Single})
        {
          for (Distribution dst_dist : {Distribution::Block,
                                        Distribution::Replicated,
                                        Distribution::Single})
          {
            ProcessorGrid grid = ProcessorGrid(comm, grid_shape);
            ShapeTuple src_tensor_shape(8, 5, 12);
            src_tensor_shape.set_size(grid.ndim());
            ShapeTuple dst_tensor_shape(2, 19, 4);
            dst_tensor_shape.set_size(grid.ndim());
            DTTuple tensor_dim_types(TuplePad<DTTuple>(grid.ndim(), DT::Any));
            DistTTuple src_tensor_dist(
                TuplePad<DistTTuple>(grid.ndim(), src_dist));
            DistTTuple dst_tensor_dist(
                TuplePad<DistTTuple>(grid.ndim(), dst_dist));
            SrcTensorType src_tensor = SrcTensorType(
              SrcDev, src_tensor_shape, tensor_dim_types, grid, src_tensor_dist);
            DstTensorType dst_tensor = DstTensorType(
              DstDev, dst_tensor_shape, tensor_dim_types, grid, dst_tensor_dist);

            for (DataIndexType i = 0; i < src_tensor.local_numel(); ++i)
            {
              write_ele<SrcDev>(src_tensor.data(), i, src_val);
            }
            for (DataIndexType i = 0; i < dst_tensor.local_numel(); ++i)
            {
              write_ele<DstDev>(dst_tensor.data(), i, dst_val);
            }

            REQUIRE_NOTHROW(copy(dst_tensor, src_tensor));

            REQUIRE(dst_tensor.shape() == src_tensor_shape);
            REQUIRE(dst_tensor.local_shape() == src_tensor.local_shape());
            REQUIRE(dst_tensor.dim_types() == tensor_dim_types);
            REQUIRE(dst_tensor.distribution() == src_tensor_dist);
            REQUIRE(dst_tensor.numel() == src_tensor.numel());
            REQUIRE_FALSE(dst_tensor.is_empty());
            REQUIRE(dst_tensor.is_local_empty() == src_tensor.is_local_empty());
            REQUIRE_FALSE(dst_tensor.is_view());
            if (src_tensor.is_local_empty())
            {
              REQUIRE(dst_tensor.data() == nullptr);
            }
            else
            {
              REQUIRE(dst_tensor.data() != src_tensor.data());
            }

            for (DataIndexType i = 0; i < src_tensor.local_numel(); ++i)
            {
              REQUIRE(read_ele<SrcDev>(src_tensor.data(), i) == src_val);
              REQUIRE(read_ele<DstDev>(dst_tensor.data(), i) == src_val);
            }
          }
        }
      }, comm);
    });
  }

  SECTION("Copying an empty tensor works")
  {
    for_comms([&](Comm& comm) {
      for_grid_shapes([&](ShapeTuple grid_shape) {
        for (Distribution dist : {Distribution::Block,
                                  Distribution::Replicated,
                                  Distribution::Single})
        {
          ProcessorGrid grid = ProcessorGrid(comm, grid_shape);
          ShapeTuple tensor_shape(8, 5, 12);
          tensor_shape.set_size(grid.ndim());
          DTTuple tensor_dim_types(TuplePad<DTTuple>(grid.ndim(), DT::Any));
          DistTTuple tensor_dist(TuplePad<DistTTuple>(grid.ndim(), dist));
          SrcTensorType src_tensor = SrcTensorType(SrcDev, grid);
          DstTensorType dst_tensor = DstTensorType(
              DstDev, tensor_shape, tensor_dim_types, grid, tensor_dist);

          REQUIRE_NOTHROW(copy(dst_tensor, src_tensor));

          REQUIRE(dst_tensor.is_empty());
        }
      }, comm);
    });
  }
}
