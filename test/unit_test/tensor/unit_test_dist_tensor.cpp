////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "h2/tensor/dist_tensor.hpp"
#include "h2/utils/typename.hpp"
#include "utils.hpp"
#include "../mpi_utils.hpp"

using namespace h2;

TEMPLATE_LIST_TEST_CASE("Distributed tensors can be created",
                        "[dist-tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using DistTensorType = DistTensor<DataType>;

  for_comms([&](Comm& comm) {
    for_grid_shapes([&](ShapeTuple shape) {
      ProcessorGrid grid = ProcessorGrid(comm, shape);
      REQUIRE_NOTHROW(DistTensorType(Dev, grid));
      REQUIRE_NOTHROW(DistTensorType(
          Dev,
          shape,
          DTTuple(TuplePad<DTTuple>(shape.size(), DT::Any)),
          grid,
          DistTTuple(TuplePad<DistTTuple>(shape.size(), Distribution::Block))));
    }, comm);
  });
}

TEMPLATE_LIST_TEST_CASE("Distributed tensor metadata is sane",
                        "[dist-tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using DistTensorType = DistTensor<DataType>;

  SECTION("Block distribution")
  {
    for_comms([&](Comm& comm) {
      for_grid_shapes([&](ShapeTuple grid_shape) {
        ProcessorGrid grid = ProcessorGrid(comm, grid_shape);
        ShapeTuple tensor_shape(8, 5, 12);
        tensor_shape.set_size(grid.ndim());
        DTTuple tensor_dim_types(TuplePad<DTTuple>(grid.ndim(), DT::Any));
        DistTTuple tensor_dist(TuplePad<DistTTuple>(grid.ndim(), Distribution::Block));
        DistTensorType tensor = DistTensorType(
            Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

        // Compute the local shape.
        ShapeTuple local_shape(TuplePad<ShapeTuple>(grid.ndim(), 0));
        for (typename ShapeTuple::size_type i = 0; i < grid.ndim(); ++i)
        {
          local_shape[i] = tensor_shape[i] / grid_shape[i];
          if (grid.get_dimension_rank(i) < tensor_shape[i] % grid_shape[i])
          {
            local_shape[i] += 1;
          }
        }
        DataIndexType local_numel = product<DataIndexType>(local_shape);
        bool is_local_empty = local_numel == 0;
        if (is_local_empty)
        {
          // If a dimension is 0, the shape is empty.
          local_shape = ShapeTuple();
        }

        REQUIRE(tensor.get_type_info() == get_h2_type<DataType>());
        REQUIRE(tensor.shape() == tensor_shape);
        REQUIRE(tensor.dim_types() == tensor_dim_types);
        REQUIRE(tensor.local_shape() == local_shape);
        REQUIRE(tensor.proc_grid() == grid);
        REQUIRE(tensor.distribution() == tensor_dist);
        for (typename ShapeTuple::size_type i = 0; i < grid_shape.size(); ++i)
        {
          REQUIRE(tensor.shape(i) == tensor_shape[i]);
          REQUIRE(tensor.dim_type(i) == tensor_dim_types[i]);
          REQUIRE(tensor.distribution(i) == tensor_dist[i]);
        }
        for (typename ShapeTuple::size_type i = 0; i < local_shape.size(); ++i)
        {
          REQUIRE(tensor.local_shape(i) == local_shape[i]);
        }
        REQUIRE(tensor.ndim() == tensor_shape.size());
        REQUIRE(tensor.numel() == product<DataIndexType>(tensor_shape));
        REQUIRE_FALSE(tensor.is_empty());
        REQUIRE(tensor.local_numel() == local_numel);
        REQUIRE(tensor.is_local_empty() == is_local_empty);
        REQUIRE_FALSE(tensor.is_view());
        REQUIRE_FALSE(tensor.is_const_view());
        REQUIRE(tensor.get_view_type() == ViewType::None);
        REQUIRE(tensor.get_device() == TestType::value);
        typename DistTensorType::local_tensor_type& local_tensor =
            tensor.local_tensor();
        REQUIRE(local_tensor.shape() == local_shape);
        if (is_local_empty)
        {
          REQUIRE(local_tensor.dim_types() == DTTuple{});
          REQUIRE(tensor.data() == nullptr);
          REQUIRE(tensor.const_data() == nullptr);
        }
        else
        {
          REQUIRE(local_tensor.dim_types() == tensor_dim_types);
          REQUIRE(tensor.data() != nullptr);
          REQUIRE(tensor.const_data() != nullptr);
        }
        REQUIRE(tensor.data() == local_tensor.data());
        REQUIRE(tensor.const_data() == local_tensor.const_data());
        REQUIRE_FALSE(tensor.is_lazy());
      }, comm, 0, 3);
    });
  }

  SECTION("Replicated distribution")
  {
    for_comms([&](Comm& comm) {
      for_grid_shapes([&](ShapeTuple grid_shape) {
        ProcessorGrid grid = ProcessorGrid(comm, grid_shape);
        ShapeTuple tensor_shape(8, 5, 12);
        tensor_shape.set_size(grid.ndim());
        DTTuple tensor_dim_types(TuplePad<DTTuple>(grid.ndim(), DT::Any));
        DistTTuple tensor_dist(TuplePad<DistTTuple>(grid.ndim(), Distribution::Replicated));
        DistTensorType tensor = DistTensorType(
            Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

        // Compute the local shape.
        ShapeTuple local_shape = tensor_shape;

        REQUIRE(tensor.get_type_info() == get_h2_type<DataType>());
        REQUIRE(tensor.shape() == tensor_shape);
        REQUIRE(tensor.dim_types() == tensor_dim_types);
        REQUIRE(tensor.local_shape() == local_shape);
        REQUIRE(tensor.proc_grid() == grid);
        REQUIRE(tensor.distribution() == tensor_dist);
        for (typename ShapeTuple::size_type i = 0; i < grid_shape.size(); ++i)
        {
          REQUIRE(tensor.shape(i) == tensor_shape[i]);
          REQUIRE(tensor.local_shape(i) == local_shape[i]);
          REQUIRE(tensor.dim_type(i) == tensor_dim_types[i]);
          REQUIRE(tensor.distribution(i) == tensor_dist[i]);
        }
        REQUIRE(tensor.ndim() == tensor_shape.size());
        REQUIRE(tensor.numel() == product<DataIndexType>(tensor_shape));
        REQUIRE_FALSE(tensor.is_empty());
        REQUIRE(tensor.local_numel() == product<DataIndexType>(local_shape));
        REQUIRE_FALSE(tensor.is_local_empty());
        REQUIRE_FALSE(tensor.is_view());
        REQUIRE_FALSE(tensor.is_const_view());
        REQUIRE(tensor.get_view_type() == ViewType::None);
        REQUIRE(tensor.get_device() == TestType::value);
        typename DistTensorType::local_tensor_type& local_tensor =
            tensor.local_tensor();
        REQUIRE(local_tensor.shape() == local_shape);
        REQUIRE(local_tensor.dim_types() == tensor_dim_types);
        REQUIRE(tensor.data() != nullptr);
        REQUIRE(tensor.const_data() != nullptr);
        REQUIRE(tensor.data() == local_tensor.data());
        REQUIRE(tensor.const_data() == local_tensor.const_data());
        REQUIRE_FALSE(tensor.is_lazy());
      }, comm, 0, 3);
    });
  }

  SECTION("Single distribution")
  {
    for_comms([&](Comm& comm) {
      for_grid_shapes([&](ShapeTuple grid_shape) {
        ProcessorGrid grid = ProcessorGrid(comm, grid_shape);
        ShapeTuple tensor_shape(8, 5, 12);
        tensor_shape.set_size(grid.ndim());
        DTTuple tensor_dim_types(TuplePad<DTTuple>(grid.ndim(), DT::Any));
        DistTTuple tensor_dist(TuplePad<DistTTuple>(grid.ndim(), Distribution::Single));
        DistTensorType tensor = DistTensorType(
            Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

        // Compute the local shape.
        ShapeTuple local_shape;
        bool is_local_empty = true;
        DataIndexType local_numel = 0;
        if (grid.rank() == 0)
        {
          local_shape = tensor_shape;
          is_local_empty = false;
          local_numel = product<DataIndexType>(local_shape);
        }

        REQUIRE(tensor.get_type_info() == get_h2_type<DataType>());
        REQUIRE(tensor.shape() == tensor_shape);
        REQUIRE(tensor.dim_types() == tensor_dim_types);
        REQUIRE(tensor.local_shape() == local_shape);
        REQUIRE(tensor.proc_grid() == grid);
        REQUIRE(tensor.distribution() == tensor_dist);
        for (typename ShapeTuple::size_type i = 0; i < tensor_shape.size(); ++i)
        {
          REQUIRE(tensor.shape(i) == tensor_shape[i]);
          REQUIRE(tensor.dim_type(i) == tensor_dim_types[i]);
          REQUIRE(tensor.distribution(i) == tensor_dist[i]);
        }
        for (typename ShapeTuple::size_type i = 0; i < local_shape.size(); ++i)
        {
          REQUIRE(tensor.local_shape(i) == local_shape[i]);
        }
        REQUIRE(tensor.ndim() == tensor_shape.size());
        REQUIRE(tensor.numel() == product<DataIndexType>(tensor_shape));
        REQUIRE_FALSE(tensor.is_empty());
        REQUIRE(tensor.local_numel() == local_numel);
        REQUIRE(tensor.is_local_empty() == is_local_empty);
        REQUIRE_FALSE(tensor.is_view());
        REQUIRE_FALSE(tensor.is_const_view());
        REQUIRE(tensor.get_view_type() == ViewType::None);
        REQUIRE(tensor.get_device() == TestType::value);
        typename DistTensorType::local_tensor_type& local_tensor =
            tensor.local_tensor();
        REQUIRE(local_tensor.shape() == local_shape);
        if (is_local_empty)
        {
          REQUIRE(local_tensor.dim_types() == DTTuple{});
          REQUIRE(tensor.data() == nullptr);
          REQUIRE(tensor.const_data() == nullptr);
        }
        else
        {
          REQUIRE(local_tensor.dim_types() == tensor_dim_types);
          REQUIRE(tensor.data() != nullptr);
          REQUIRE(tensor.const_data() != nullptr);
        }
        REQUIRE(tensor.data() == local_tensor.data());
        REQUIRE(tensor.const_data() == local_tensor.const_data());
        REQUIRE_FALSE(tensor.is_lazy());
      }, comm, 0, 3);
    });
  }

  SECTION("Block x replicated x single distribution")
  {
    for_comms([&](Comm& comm) {
      for_grid_shapes([&](ShapeTuple grid_shape) {
        ProcessorGrid grid = ProcessorGrid(comm, grid_shape);
        ShapeTuple tensor_shape(8, 5, 12);
        DTTuple tensor_dim_types(DT::Any, DT::Any, DT::Any);
        DistTTuple tensor_dist(Distribution::Block,
                               Distribution::Replicated,
                               Distribution::Single);
        DistTensorType tensor = DistTensorType(
            Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

        // Compute the local shape.
        ShapeTuple local_shape(TuplePad<ShapeTuple>(grid.ndim()));
        // Only the roots of the third dimension may have data.
        if (grid.get_dimension_rank(2) == 0)
        {
          local_shape[0] = tensor_shape[0] / grid.shape(0);
          if (grid.get_dimension_rank(0) < tensor_shape[0] % grid.shape(0))
          {
            local_shape[0] += 1;
          }
          if (local_shape[0] != 0)
          {
            local_shape[1] = tensor_shape[1];
            local_shape[2] = tensor_shape[2];
          }
        }
        DataIndexType local_numel = product<DataIndexType>(local_shape);
        bool is_local_empty = local_numel == 0;
        if (is_local_empty)
        {
          local_shape = ShapeTuple{};
        }

        REQUIRE(tensor.get_type_info() == get_h2_type<DataType>());
        REQUIRE(tensor.shape() == tensor_shape);
        REQUIRE(tensor.dim_types() == tensor_dim_types);
        REQUIRE(tensor.local_shape() == local_shape);
        REQUIRE(tensor.proc_grid() == grid);
        REQUIRE(tensor.distribution() == tensor_dist);
        for (typename ShapeTuple::size_type i = 0; i < grid.ndim(); ++i)
        {
          REQUIRE(tensor.shape(i) == tensor_shape[i]);
          REQUIRE(tensor.dim_type(i) == tensor_dim_types[i]);
          REQUIRE(tensor.distribution(i) == tensor_dist[i]);
        }
        for (typename ShapeTuple::size_type i = 0; i < local_shape.size(); ++i)
        {
          REQUIRE(tensor.local_shape(i) == local_shape[i]);
        }
        REQUIRE(tensor.ndim() == tensor_shape.size());
        REQUIRE(tensor.numel() == product<DataIndexType>(tensor_shape));
        REQUIRE_FALSE(tensor.is_empty());
        REQUIRE(tensor.local_numel() == local_numel);
        REQUIRE(tensor.is_local_empty() == is_local_empty);
        REQUIRE_FALSE(tensor.is_view());
        REQUIRE_FALSE(tensor.is_const_view());
        REQUIRE(tensor.get_view_type() == ViewType::None);
        REQUIRE(tensor.get_device() == TestType::value);
        typename DistTensorType::local_tensor_type& local_tensor =
            tensor.local_tensor();
        REQUIRE(local_tensor.shape() == local_shape);
        if (is_local_empty)
        {
          REQUIRE(local_tensor.dim_types() == DTTuple{});
          REQUIRE(tensor.data() == nullptr);
          REQUIRE(tensor.const_data() == nullptr);
        }
        else
        {
          REQUIRE(local_tensor.dim_types() == tensor_dim_types);
          REQUIRE(tensor.data() != nullptr);
          REQUIRE(tensor.const_data() != nullptr);
        }
        REQUIRE(tensor.data() == local_tensor.data());
        REQUIRE(tensor.const_data() == local_tensor.const_data());
        REQUIRE_FALSE(tensor.is_lazy());
      }, comm, 3, 3);
    });
  }
}

TEMPLATE_LIST_TEST_CASE("Base distributed tensor metadata is sane",
                        "[dist-tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using DistTensorType = DistTensor<DataType>;

  Comm& comm = get_comm_or_skip(1);
  ProcessorGrid grid = ProcessorGrid(comm, ShapeTuple{1});
  std::unique_ptr<BaseDistTensor> tensor =
      std::make_unique<DistTensorType>(Dev,
                                       ShapeTuple{8},
                                       DTTuple{DT::Any},
                                       grid,
                                       DistTTuple{Distribution::Block});
  REQUIRE(tensor->get_type_info() == get_h2_type<DataType>());
  REQUIRE(tensor->shape() == ShapeTuple{8});
  REQUIRE(tensor->dim_types() == DTTuple{DT::Any});
  REQUIRE(tensor->local_shape() == ShapeTuple{8});
  REQUIRE(tensor->proc_grid() == grid);
  REQUIRE(tensor->distribution() == DistTTuple{Distribution::Block});
  REQUIRE(tensor->shape(0) == 8);
  REQUIRE(tensor->dim_type(0) == DT::Any);
  REQUIRE(tensor->distribution(0) == Distribution::Block);
  REQUIRE(tensor->local_shape(0) == 8);
  REQUIRE(tensor->ndim() == 1);
  REQUIRE(tensor->numel() == 8);
  REQUIRE_FALSE(tensor->is_empty());
  REQUIRE(tensor->local_numel() == 8);
  REQUIRE_FALSE(tensor->is_local_empty());
  REQUIRE_FALSE(tensor->is_view());
  REQUIRE_FALSE(tensor->is_const_view());
  REQUIRE(tensor->get_view_type() == ViewType::None);
  REQUIRE(tensor->get_device() == TestType::value);
}

TEMPLATE_LIST_TEST_CASE("Empty distributed tensor metadata is sane",
                        "[dist-tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using DistTensorType = DistTensor<DataType>;

  for_comms([&](Comm& comm) {
    for_grid_shapes([&](ShapeTuple grid_shape) {
      ProcessorGrid grid = ProcessorGrid(comm, grid_shape);
      DistTensorType tensor = DistTensorType(Dev, grid);

      REQUIRE(tensor.get_type_info() == get_h2_type<DataType>());
      REQUIRE(tensor.shape() == ShapeTuple{});
      REQUIRE(tensor.dim_types() == DTTuple{});
      REQUIRE(tensor.distribution() == DistTTuple{});
      REQUIRE(tensor.local_shape() == ShapeTuple{});
      REQUIRE(tensor.proc_grid() == grid);
      REQUIRE(tensor.ndim() == 0);
      REQUIRE(tensor.numel() == 0);
      REQUIRE(tensor.is_empty());
      REQUIRE(tensor.local_numel() == 0);
      REQUIRE(tensor.is_local_empty());
      REQUIRE_FALSE(tensor.is_view());
      REQUIRE_FALSE(tensor.is_const_view());
      REQUIRE(tensor.get_view_type() == ViewType::None);
      REQUIRE(tensor.get_device() == TestType::value);
      typename DistTensorType::local_tensor_type& local_tensor =
          tensor.local_tensor();
      REQUIRE(local_tensor.shape() == ShapeTuple{});
      REQUIRE(local_tensor.dim_types() == DTTuple{});
      REQUIRE(local_tensor.is_empty());
      REQUIRE(tensor.data() == nullptr);
      REQUIRE(tensor.const_data() == nullptr);
      REQUIRE_FALSE(tensor.is_lazy());
    }, comm);
  });
}

TEMPLATE_LIST_TEST_CASE("Lazy and strict distributed tensors are sane",
                        "[dist-tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using DistTensorType = DistTensor<DataType>;

  Comm& comm = get_comm_or_skip(1);
  ProcessorGrid grid(comm, ShapeTuple{1});

  REQUIRE_FALSE(DistTensorType(Dev, grid).is_lazy());
  REQUIRE_FALSE(DistTensorType(Dev, grid, StrictAlloc).is_lazy());
  REQUIRE(DistTensorType(Dev, grid, LazyAlloc).is_lazy());

  REQUIRE_FALSE(
    DistTensorType(Dev, {4}, {DT::Any}, grid, {Distribution::Block}, StrictAlloc)
          .is_lazy());
  REQUIRE(DistTensorType(Dev, {4}, {DT::Any}, grid, {Distribution::Block}, LazyAlloc)
              .is_lazy());
}

TEMPLATE_LIST_TEST_CASE("Resizing distributed tensors works",
                        "[dist-tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using DistTensorType = DistTensor<DataType>;

  SECTION("Resizing works")
  {
    for_comms([&](Comm& comm) {
      for_grid_shapes([&](ShapeTuple grid_shape) {
        ProcessorGrid grid = ProcessorGrid(comm, grid_shape);

        ShapeTuple tensor_shape(8, 5);
        tensor_shape.set_size(grid.ndim());
        DTTuple tensor_dim_types(TuplePad<DTTuple>(grid.ndim(), DT::Any));
        DistTTuple tensor_dist(TuplePad<DistTTuple>(grid.ndim(), Distribution::Block));
        DistTensorType tensor = DistTensorType(
            Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

        ShapeTuple new_shape(4, 10);
        new_shape.set_size(grid.ndim());
        ShapeTuple new_local_shape(TuplePad<ShapeTuple>(grid.ndim(), 0));
        for (typename ShapeTuple::size_type i = 0; i < grid.ndim(); ++i)
        {
          new_local_shape[i] = new_shape[i] / grid.shape(i);
          if (grid.get_dimension_rank(i) < new_shape[i] % grid.shape(i))
          {
            new_local_shape[i] += 1;
          }
        }
        DataIndexType new_local_numel = product<DataIndexType>(new_local_shape);
        bool is_local_empty = new_local_numel == 0;
        if (is_local_empty)
        {
          new_local_shape = ShapeTuple();
        }

        tensor.resize(new_shape);
        REQUIRE(tensor.shape() == new_shape);
        REQUIRE(tensor.dim_types() == tensor_dim_types);
        REQUIRE(tensor.distribution() == tensor_dist);
        REQUIRE(tensor.proc_grid() == grid);
        REQUIRE(tensor.ndim() == new_shape.size());
        REQUIRE(tensor.numel() == product<DataIndexType>(new_shape));
        REQUIRE_FALSE(tensor.is_empty());
        REQUIRE(tensor.local_numel() == new_local_numel);
        REQUIRE(tensor.is_local_empty() == is_local_empty);
        typename DistTensorType::local_tensor_type& local_tensor =
          tensor.local_tensor();
        REQUIRE(local_tensor.shape() == new_local_shape);
        if (is_local_empty)
        {
          REQUIRE(local_tensor.dim_types() == DTTuple{});
          REQUIRE(tensor.data() == nullptr);
          REQUIRE(tensor.const_data() == nullptr);
        }
        else
        {
          REQUIRE(local_tensor.dim_types() == tensor_dim_types);
          REQUIRE(tensor.data() != nullptr);
          REQUIRE(tensor.const_data() != nullptr);
        }
      }, comm, 0, 2);
    });
  }

  SECTION("Emptying tensors")
  {
    for_comms([&](Comm& comm) {
      for_grid_shapes([&](ShapeTuple grid_shape) {
        ProcessorGrid grid = ProcessorGrid(comm, grid_shape);

        ShapeTuple tensor_shape(8, 5);
        tensor_shape.set_size(grid.ndim());
        DTTuple tensor_dim_types(TuplePad<DTTuple>(grid.ndim(), DT::Any));
        DistTTuple tensor_dist(TuplePad<DistTTuple>(grid.ndim(), Distribution::Block));
        DistTensorType tensor = DistTensorType(
            Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

        tensor.empty();
        REQUIRE(tensor.shape() == ShapeTuple{});
        REQUIRE(tensor.dim_types() == DTTuple{});
        REQUIRE(tensor.distribution() == DistTTuple{});
        REQUIRE(tensor.proc_grid() == grid);
        REQUIRE(tensor.ndim() == 0);
        REQUIRE(tensor.numel() == 0);
        REQUIRE(tensor.is_empty());
        REQUIRE(tensor.local_numel() == 0);
        REQUIRE(tensor.is_local_empty());
        typename DistTensorType::local_tensor_type& local_tensor =
          tensor.local_tensor();
        REQUIRE(local_tensor.shape() == ShapeTuple{});
        REQUIRE(local_tensor.dim_types() == DTTuple{});
        REQUIRE(tensor.data() == nullptr);
        REQUIRE(tensor.const_data() == nullptr);
      }, comm, 0, 2);
    });
  }

  SECTION("Emptying then resizing tensors")
  {
    for_comms([&](Comm& comm) {
      for_grid_shapes([&](ShapeTuple grid_shape) {
        ProcessorGrid grid = ProcessorGrid(comm, grid_shape);

        ShapeTuple tensor_shape(8, 5);
        tensor_shape.set_size(grid.ndim());
        DTTuple tensor_dim_types(TuplePad<DTTuple>(grid.ndim(), DT::Any));
        DistTTuple tensor_dist(TuplePad<DistTTuple>(grid.ndim(), Distribution::Block));
        DistTensorType tensor = DistTensorType(
            Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

        tensor.empty();
        ShapeTuple new_shape(4, 10);
        new_shape.set_size(grid.ndim());
        ShapeTuple new_local_shape(TuplePad<ShapeTuple>(grid.ndim(), 0));
        for (typename ShapeTuple::size_type i = 0; i < grid.ndim(); ++i)
        {
          new_local_shape[i] = new_shape[i] / grid.shape(i);
          if (grid.get_dimension_rank(i) < new_shape[i] % grid.shape(i))
          {
            new_local_shape[i] += 1;
          }
        }
        DataIndexType new_local_numel = product<DataIndexType>(new_local_shape);
        bool is_local_empty = new_local_numel == 0;
        if (is_local_empty)
        {
          new_local_shape = ShapeTuple();
        }

        tensor.resize(new_shape, tensor_dim_types, tensor_dist);
        REQUIRE(tensor.shape() == new_shape);
        REQUIRE(tensor.dim_types() == tensor_dim_types);
        REQUIRE(tensor.distribution() == tensor_dist);
        REQUIRE(tensor.proc_grid() == grid);
        REQUIRE(tensor.ndim() == new_shape.size());
        REQUIRE(tensor.numel() == product<DataIndexType>(new_shape));
        REQUIRE_FALSE(tensor.is_empty());
        REQUIRE(tensor.local_numel() == new_local_numel);
        REQUIRE(tensor.is_local_empty() == is_local_empty);
        typename DistTensorType::local_tensor_type& local_tensor =
          tensor.local_tensor();
        REQUIRE(local_tensor.shape() == new_local_shape);
        if (is_local_empty)
        {
          REQUIRE(local_tensor.dim_types() == DTTuple{});
          REQUIRE(tensor.data() == nullptr);
          REQUIRE(tensor.const_data() == nullptr);
        }
        else
        {
          REQUIRE(local_tensor.dim_types() == tensor_dim_types);
          REQUIRE(tensor.data() != nullptr);
          REQUIRE(tensor.const_data() != nullptr);
        }
      }, comm, 0, 2);
    });
  }
}

TEMPLATE_LIST_TEST_CASE("Writing to distributed tensors works",
                        "[dist-tensor]",
                        AllDevList)
{
  using DistTensorType = DistTensor<DataType>;
  constexpr Device Dev = TestType::value;

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
        DistTensorType tensor = DistTensorType(
            Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);
        typename DistTensorType::local_tensor_type& local_tensor =
          tensor.local_tensor();

        DataType* buf = tensor.data();
        for (DataIndexType i = 0; i < tensor.local_numel(); ++i)
        {
          write_ele<Dev>(buf, i, static_cast<DataType>(i), tensor.get_stream());
        }

        DataIndexType idx = 0;
        for_ndim(local_tensor.shape(), [&](ScalarIndexTuple c) {
          REQUIRE(read_ele<Dev>(local_tensor.get(c), tensor.get_stream())
                  == idx);
          ++idx;
        });
      }
    }, comm, 0, 3);
  });
}

TEMPLATE_LIST_TEST_CASE(
    "Attaching distributed tensors to existing buffers works",
    "[dist-tensor]",
    AllDevList)
{
  using DistTensorType = DistTensor<DataType>;
  constexpr Device Dev = TestType::value;

  for_comms([&](Comm& comm) {
    for_grid_shapes([&](ShapeTuple grid_shape) {
      ProcessorGrid grid = ProcessorGrid(comm, grid_shape);

      ShapeTuple tensor_local_shape(8, 5, 12);
      tensor_local_shape.set_size(grid.ndim());
      StrideTuple tensor_local_strides(1, 8, 40);
      tensor_local_strides.set_size(grid.ndim());
      ShapeTuple tensor_shape = tensor_local_shape;
      for (typename ShapeTuple::size_type i = 0; i < tensor_shape.size(); ++i)
      {
        tensor_shape[i] *= grid.shape(i);
      }
      DTTuple tensor_dim_types(TuplePad<DTTuple>(grid.ndim(), DT::Any));
      DistTTuple tensor_dist(TuplePad<DistTTuple>(grid.ndim(), Distribution::Block));
      DataIndexType buf_size = product<DataIndexType>(tensor_local_shape);

      DeviceBuf<DataType, Dev> buf(buf_size);
      for (DataIndexType i = 0; i < buf_size; ++i)
      {
        write_ele<Dev>(
          buf.buf, i, static_cast<DataType>(i), ComputeStream{Dev});
      }

      DistTensorType tensor(Dev,
                            buf.buf,
                            tensor_shape,
                            tensor_dim_types,
                            grid,
                            tensor_dist,
                            tensor_local_shape,
                            tensor_local_strides,
                            ComputeStream{Dev});

      REQUIRE(tensor.shape() == tensor_shape);
      REQUIRE(tensor.dim_types() == tensor_dim_types);
      REQUIRE(tensor.distribution() == tensor_dist);
      REQUIRE(tensor.proc_grid() == grid);
      REQUIRE(tensor.local_shape() == tensor_local_shape);
      REQUIRE(tensor.local_numel() == buf_size);
      REQUIRE(tensor.ndim() == grid.ndim());
      REQUIRE(tensor.numel() == grid.size() * buf_size);
      REQUIRE(tensor.is_view());
      REQUIRE_FALSE(tensor.is_const_view());
      REQUIRE(tensor.get_view_type() == ViewType::Mutable);
      REQUIRE(tensor.data() == buf.buf);

      typename DistTensorType::local_tensor_type& local_tensor =
          tensor.local_tensor();
      REQUIRE(local_tensor.shape() == tensor_local_shape);
      REQUIRE(local_tensor.strides() == tensor_local_strides);
      for (DataIndexType i = 0; i < local_tensor.numel(); ++i)
      {
        REQUIRE(read_ele<Dev>(local_tensor.data(), i, tensor.get_stream())
                == i);
      }
      DataIndexType idx = 0;
      for_ndim(local_tensor.shape(), [&](ScalarIndexTuple c) {
        REQUIRE(read_ele<Dev>(local_tensor.get(c), tensor.get_stream())
                == idx);
        ++idx;
      });
    }, comm, 0, 3);
  });
}

TEMPLATE_LIST_TEST_CASE("Viewing distributed tensors works",
                        "[dist-tensor]",
                        AllDevList)
{
  using DistTensorType = DistTensor<DataType>;
  constexpr Device Dev = TestType::value;

  SECTION("Basic views work")
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
          DistTensorType tensor = DistTensorType(
              Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);
          typename DistTensorType::local_tensor_type& local_tensor =
            tensor.local_tensor();

          DataType* buf = tensor.data();
          for (DataIndexType i = 0; i < tensor.local_numel(); ++i)
          {
            write_ele<Dev>(
              buf, i, static_cast<DataType>(i), tensor.get_stream());
          }

          std::unique_ptr<DistTensorType> view = tensor.view();
          REQUIRE(view->shape() == tensor_shape);
          REQUIRE(view->local_shape() == tensor.local_shape());
          REQUIRE(view->dim_types() == tensor_dim_types);
          REQUIRE(view->distribution() == tensor_dist);
          REQUIRE(view->ndim() == tensor.ndim());
          REQUIRE(view->numel() == tensor.numel());
          REQUIRE(view->is_view());
          REQUIRE_FALSE(view->is_const_view());
          REQUIRE(view->get_view_type() == ViewType::Mutable);
          REQUIRE(view->data() == tensor.data());

          typename DistTensorType::local_tensor_type& local_view =
              view->local_tensor();
          REQUIRE(local_view.shape() == local_tensor.shape());
          REQUIRE(local_view.dim_types() == local_tensor.dim_types());
          REQUIRE(local_view.strides() == local_tensor.strides());
          REQUIRE(local_view.is_view());
          REQUIRE_FALSE(local_view.is_const_view());
          REQUIRE(local_view.get_view_type() == ViewType::Mutable);
          REQUIRE(local_view.data() == local_tensor.data());

          for (DataIndexType i = 0; i < view->local_numel(); ++i)
          {
            REQUIRE(read_ele<Dev>(view->data(), i, view->get_stream()) == i);
          }
        }
      }, comm, 0, 3);
    });
  }

  SECTION("Constant views work")
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
          DistTensorType tensor = DistTensorType(
              Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);
          typename DistTensorType::local_tensor_type& local_tensor =
            tensor.local_tensor();

          DataType* buf = tensor.data();
          for (DataIndexType i = 0; i < tensor.local_numel(); ++i)
          {
            write_ele<Dev>(
              buf, i, static_cast<DataType>(i), tensor.get_stream());
          }

          std::unique_ptr<DistTensorType> view = tensor.const_view();
          REQUIRE(view->shape() == tensor_shape);
          REQUIRE(view->local_shape() == tensor.local_shape());
          REQUIRE(view->dim_types() == tensor_dim_types);
          REQUIRE(view->distribution() == tensor_dist);
          REQUIRE(view->ndim() == tensor.ndim());
          REQUIRE(view->numel() == tensor.numel());
          REQUIRE(view->is_view());
          REQUIRE(view->is_const_view());
          REQUIRE(view->get_view_type() == ViewType::Const);
          REQUIRE(view->const_data() == tensor.data());

          typename DistTensorType::local_tensor_type& local_view =
              view->local_tensor();
          REQUIRE(local_view.shape() == local_tensor.shape());
          REQUIRE(local_view.dim_types() == local_tensor.dim_types());
          REQUIRE(local_view.strides() == local_tensor.strides());
          REQUIRE(local_view.is_view());
          REQUIRE(local_view.is_const_view());
          REQUIRE(local_view.get_view_type() == ViewType::Const);
          REQUIRE(local_view.const_data() == local_tensor.data());

          for (DataIndexType i = 0; i < view->local_numel(); ++i)
          {
            REQUIRE(read_ele<Dev>(view->const_data(), i, view->get_stream())
                    == i);
          }
        }
      }, comm, 0, 3);
    });
  }

  SECTION("Viewing a subtensor works")
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
          DistTensorType tensor = DistTensorType(
              Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);
          typename DistTensorType::local_tensor_type& local_tensor =
            tensor.local_tensor();

          DataType* buf = tensor.data();
          for (DataIndexType i = 0; i < tensor.local_numel(); ++i)
          {
            write_ele<Dev>(
              buf, i, static_cast<DataType>(i), tensor.get_stream());
          }

          IndexRangeTuple view_indices({IRng(0, 4), IRng(2, 4), IRng(1, 2)});
          view_indices.set_size(grid.ndim());
          ShapeTuple view_shape = ShapeTuple{4, 2, 1};
          view_shape.set_size(grid.ndim());

          ShapeTuple local_shape;
          bool is_local_empty = false;
          IndexRangeTuple local_view_indices;
          if (dist == Distribution::Block)
          {
            IndexRangeTuple global_indices = h2::internal::get_global_indices(
                tensor_shape, grid, tensor_dist);
            if (do_index_ranges_intersect(global_indices, view_indices))
            {
              IndexRangeTuple global_view_indices_present =
                  intersect_index_ranges(global_indices, view_indices);
              local_view_indices = h2::internal::global2local_indices(
                  tensor_shape, grid, tensor_dist, global_view_indices_present);
              local_shape =
                  get_index_range_shape(local_view_indices, view_shape);
            }
            else
            {
              is_local_empty = true;
            }
          }
          else if (dist == Distribution::Replicated)
          {
            local_shape = view_shape;
            local_view_indices = view_indices;
          }
          else if (dist == Distribution::Single)
          {
            if (grid.rank() == 0)
            {
              local_shape = view_shape;
              local_view_indices = view_indices;
            }
            else
            {
              is_local_empty = true;
            }
          }
          DataIndexType local_numel = 0;
          ScalarIndexTuple local_start;
          if (!is_local_empty)
          {
            local_numel = product<DataIndexType>(local_shape);
            local_start = get_index_range_start(local_view_indices);
          }

          std::unique_ptr<DistTensorType> view = tensor.view(view_indices);
          REQUIRE(view->shape() == view_shape);
          REQUIRE(view->local_shape() == local_shape);
          REQUIRE(view->dim_types() == tensor_dim_types);
          REQUIRE(view->distribution() == tensor_dist);
          REQUIRE(view->ndim() == tensor.ndim());
          REQUIRE(view->numel() == product<DataIndexType>(view_shape));
          REQUIRE(view->local_numel() == local_numel);
          REQUIRE(view->is_view());
          REQUIRE_FALSE(view->is_const_view());
          REQUIRE(view->get_view_type() == ViewType::Mutable);
          if (is_local_empty)
          {
            REQUIRE(view->data() == nullptr);
          }
          else
          {
            REQUIRE(view->data() == local_tensor.get(local_start));
          }

          typename DistTensorType::local_tensor_type& local_view =
              view->local_tensor();
          REQUIRE(local_view.shape() == local_shape);
          REQUIRE(local_view.numel() == local_numel);
          if (is_local_empty)
          {
            REQUIRE(local_view.dim_types() == DTTuple{});
            REQUIRE(local_view.strides() == StrideTuple{});
            REQUIRE(local_view.data() == nullptr);
          }
          else
          {
            REQUIRE(local_view.dim_types() == local_tensor.dim_types());
            // 1x1x...x1 tensors always have strides 1x1x...x1.
            if (local_numel == 1)
            {
              REQUIRE(local_view.strides()
                      == StrideTuple(TuplePad<StrideTuple>(grid.ndim(), 1)));
            }
            else
            {
              REQUIRE(local_view.strides() == local_tensor.strides());
            }
            REQUIRE(local_view.data() == local_tensor.get(local_start));
          }
          REQUIRE(local_view.is_view());
          REQUIRE_FALSE(local_view.is_const_view());
          REQUIRE(local_view.get_view_type() == ViewType::Mutable);

          // TODO: Check actual data.
        }
      }, comm, 0, 3);
    });
  }

  SECTION("Eliminating dimensions fails")
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
          DistTensorType tensor = DistTensorType(
              Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

          IndexRangeTuple view_indices({IRng(0)});

          REQUIRE_THROWS(tensor.view(view_indices));
        }
      }, comm, 0, 3);
    });
  }

  SECTION("Viewing a view works")
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
          DistTensorType tensor = DistTensorType(
              Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);
          typename DistTensorType::local_tensor_type& local_tensor =
            tensor.local_tensor();

          DataType* buf = tensor.data();
          for (DataIndexType i = 0; i < tensor.local_numel(); ++i)
          {
            write_ele<Dev>(
              buf, i, static_cast<DataType>(i), tensor.get_stream());
          }

          std::unique_ptr<DistTensorType> orig_view = tensor.view();
          std::unique_ptr<DistTensorType> view = orig_view->view();
          REQUIRE(view->shape() == tensor_shape);
          REQUIRE(view->local_shape() == tensor.local_shape());
          REQUIRE(view->dim_types() == tensor_dim_types);
          REQUIRE(view->distribution() == tensor_dist);
          REQUIRE(view->ndim() == tensor.ndim());
          REQUIRE(view->numel() == tensor.numel());
          REQUIRE(view->is_view());
          REQUIRE_FALSE(view->is_const_view());
          REQUIRE(view->get_view_type() == ViewType::Mutable);
          REQUIRE(view->data() == tensor.data());

          typename DistTensorType::local_tensor_type& local_view =
              view->local_tensor();
          REQUIRE(local_view.shape() == local_tensor.shape());
          REQUIRE(local_view.dim_types() == local_tensor.dim_types());
          REQUIRE(local_view.strides() == local_tensor.strides());
          REQUIRE(local_view.is_view());
          REQUIRE_FALSE(local_view.is_const_view());
          REQUIRE(local_view.get_view_type() == ViewType::Mutable);
          REQUIRE(local_view.data() == local_tensor.data());

          for (DataIndexType i = 0; i < view->local_numel(); ++i)
          {
            REQUIRE(read_ele<Dev>(view->data(), i, view->get_stream()) == i);
          }
        }
      }, comm, 0, 3);
    });
  }

  SECTION("Unviewing a view works")
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
          DistTensorType tensor = DistTensorType(
              Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

          DataType* buf = tensor.data();
          for (DataIndexType i = 0; i < tensor.local_numel(); ++i)
          {
            write_ele<Dev>(
              buf, i, static_cast<DataType>(i), tensor.get_stream());
          }

          std::unique_ptr<DistTensorType> view = tensor.view();
          REQUIRE(view->is_view());
          view->unview();
          REQUIRE_FALSE(view->is_view());
          REQUIRE(view->shape() == ShapeTuple{});
          REQUIRE(view->local_shape() == ShapeTuple{});
          REQUIRE(view->dim_types() == DTTuple{});
          REQUIRE(view->distribution() == DistTTuple{});
          REQUIRE(view->ndim() == 0);
          REQUIRE(view->numel() == 0);
          REQUIRE(view->is_empty());
          REQUIRE(view->data() == nullptr);
        }
      }, comm, 0, 3);
    });
  }

  SECTION("Emptying a view unviews")
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
          DistTensorType tensor = DistTensorType(
              Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

          DataType* buf = tensor.data();
          for (DataIndexType i = 0; i < tensor.local_numel(); ++i)
          {
            write_ele<Dev>(
              buf, i, static_cast<DataType>(i), tensor.get_stream());
          }

          std::unique_ptr<DistTensorType> view = tensor.view();
          REQUIRE(view->is_view());
          view->empty();
          REQUIRE_FALSE(view->is_view());
          REQUIRE(view->shape() == ShapeTuple{});
          REQUIRE(view->local_shape() == ShapeTuple{});
          REQUIRE(view->dim_types() == DTTuple{});
          REQUIRE(view->distribution() == DistTTuple{});
          REQUIRE(view->ndim() == 0);
          REQUIRE(view->numel() == 0);
          REQUIRE(view->is_empty());
          REQUIRE(view->data() == nullptr);
        }
      }, comm, 0, 3);
    });
  }
}

TEMPLATE_LIST_TEST_CASE("Empty distributed tensor views work",
                        "[dist-tensor]",
                        AllDevList)
{
  using DistTensorType = DistTensor<DataType>;
  constexpr Device Dev = TestType::value;

  SECTION("View with fully empty coordinates works")
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
          DistTensorType tensor = DistTensorType(
              Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

          std::unique_ptr<DistTensorType> view = tensor.view(IndexRangeTuple{});
          REQUIRE(view->shape() == ShapeTuple{});
          REQUIRE(view->local_shape() == ShapeTuple{});
          REQUIRE(view->dim_types() == DTTuple{});
          REQUIRE(view->distribution() == DistTTuple{});
          REQUIRE(view->ndim() == 0);
          REQUIRE(view->numel() == 0);
          REQUIRE(view->is_view());
          REQUIRE_FALSE(view->is_const_view());
          REQUIRE(view->get_view_type() == ViewType::Mutable);
          REQUIRE(view->data() == nullptr);

          typename DistTensorType::local_tensor_type& local_view =
              view->local_tensor();
          REQUIRE(local_view.shape() == ShapeTuple{});
          REQUIRE(local_view.dim_types() == DTTuple{});
          REQUIRE(local_view.strides() == StrideTuple{});
          REQUIRE(local_view.is_view());
          REQUIRE_FALSE(local_view.is_const_view());
          REQUIRE(local_view.get_view_type() == ViewType::Mutable);
          REQUIRE(local_view.data() == nullptr);
        }
      }, comm, 0, 3);
    });
  }

  SECTION("View with one coordinate empty works")
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
          DistTensorType tensor = DistTensorType(
              Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

          IndexRangeTuple view_indices({IRng(), IRng(2, 3), IRng(1, 2)});
          view_indices.set_size(grid.ndim());

          std::unique_ptr<DistTensorType> view = tensor.view(view_indices);
          REQUIRE(view->shape() == ShapeTuple{});
          REQUIRE(view->local_shape() == ShapeTuple{});
          REQUIRE(view->dim_types() == DTTuple{});
          REQUIRE(view->distribution() == DistTTuple{});
          REQUIRE(view->ndim() == 0);
          REQUIRE(view->numel() == 0);
          REQUIRE(view->is_view());
          REQUIRE_FALSE(view->is_const_view());
          REQUIRE(view->get_view_type() == ViewType::Mutable);
          REQUIRE(view->data() == nullptr);

          typename DistTensorType::local_tensor_type& local_view =
              view->local_tensor();
          REQUIRE(local_view.shape() == ShapeTuple{});
          REQUIRE(local_view.dim_types() == DTTuple{});
          REQUIRE(local_view.strides() == StrideTuple{});
          REQUIRE(local_view.is_view());
          REQUIRE_FALSE(local_view.is_const_view());
          REQUIRE(local_view.get_view_type() == ViewType::Mutable);
          REQUIRE(local_view.data() == nullptr);
        }
      }, comm, 0, 3);
    });
  }
}

TEMPLATE_LIST_TEST_CASE("Cloning distributed tensors works",
                        "[dist-tensor]",
                        AllDevList)
{
  using DistTensorType = DistTensor<DataType>;
  constexpr Device Dev = TestType::value;

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
        DistTensorType tensor = DistTensorType(
            Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);
        std::unique_ptr<DistTensorType> clone = tensor.clone();

        REQUIRE(clone->shape() == tensor.shape());
        REQUIRE(clone->dim_types() == tensor.dim_types());
        REQUIRE(clone->distribution() == tensor.distribution());
        REQUIRE(clone->local_shape() == tensor.local_shape());
        REQUIRE(clone->proc_grid() == tensor.proc_grid());
        REQUIRE(clone->ndim() == tensor.ndim());
        REQUIRE(clone->numel() == tensor.numel());
        REQUIRE(clone->is_empty() == tensor.is_empty());
        REQUIRE(clone->local_numel() == tensor.local_numel());
        REQUIRE(clone->is_local_empty() == tensor.is_local_empty());
        REQUIRE_FALSE(clone->is_view());
        REQUIRE_FALSE(clone->is_const_view());
        REQUIRE(clone->get_view_type() == ViewType::None);
        REQUIRE(clone->get_device() == tensor.get_device());
        REQUIRE(clone->is_lazy() == tensor.is_lazy());
        if (tensor.is_local_empty())
        {
          REQUIRE(clone->data() == nullptr);
        }
        else
        {
          REQUIRE(clone->data() != nullptr);
          REQUIRE(clone->data() != tensor.data());
        }
      }
    }, comm, 0, 3);
  });
}

TEMPLATE_LIST_TEST_CASE("DistTensor get/set stream works",
                        "[dist-tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using DistTensorType = DistTensor<DataType>;

  Comm& comm = get_comm_or_skip(1);
  ProcessorGrid grid(comm, ShapeTuple{1});

  ComputeStream stream1 = create_new_compute_stream<Dev>();
  ComputeStream stream2 = create_new_compute_stream<Dev>();

  SECTION("Get/set on regular tensor")
  {
    DistTensorType tensor{Dev,
                          ShapeTuple{1},
                          DTTuple{DT::Any},
                          grid,
                          DistTTuple{Distribution::Block},
                          StrictAlloc,
                          stream1};
    REQUIRE(tensor.get_stream() == stream1);
    tensor.set_stream(stream2);
    REQUIRE(tensor.get_stream() == stream2);
  }

  SECTION("Get/set on view")
  {
    DistTensorType tensor{Dev,
                          ShapeTuple{1},
                          DTTuple{DT::Any},
                          grid,
                          DistTTuple{Distribution::Block},
                          StrictAlloc,
                          stream1};
    auto view = tensor.view();
    ComputeStream stream3 = create_new_compute_stream<Dev>();
    // Changing the original should not impact the view.
    REQUIRE(tensor.get_stream() == stream1);
    REQUIRE(view->get_stream() == stream1);
    tensor.set_stream(stream2);
    REQUIRE(tensor.get_stream() == stream2);
    REQUIRE(view->get_stream() == stream1);
    view->set_stream(stream3);
    REQUIRE(view->get_stream() == stream3);
    REQUIRE(tensor.get_stream() == stream2);
  }
}

TEMPLATE_LIST_TEST_CASE("Distributed tensors are printable",
                        "[dist-tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using DistTensorType = DistTensor<DataType>;

  std::stringstream dev_ss;
  dev_ss << TestType::value;

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
        DistTensorType tensor = DistTensorType(
            Dev, tensor_shape, tensor_dim_types, grid, tensor_dist);

        std::stringstream ss;
        ss << tensor;

        // Not testing this exactly since the output gets complicated.
        REQUIRE_THAT(ss.str(),
                     Catch::Matchers::StartsWith(std::string("DistTensor<")
                                                 + TypeName<DataType>() + ", "
                                                 + dev_ss.str() + ">"));
      }
    }, comm, 0, 3);
  });
}
