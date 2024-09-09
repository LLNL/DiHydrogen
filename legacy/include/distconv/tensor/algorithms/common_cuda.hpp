#pragma once

#include "distconv/tensor/tensor.hpp"

#include <type_traits>

namespace distconv
{
namespace tensor
{
namespace algorithms_cuda
{

constexpr int DEFAULT_BLOCK_SIZE = 256;
constexpr int DEFAULT_MAX_THREAD_WORK_SIZE = 8;

template <int BLOCK_SIZE, int MAX_THREAD_WORK_SIZE>
void get_grid_dims(Shape const& region, dim3& grid_dims, int& thread_work_size)
{
  int const nd = region.num_dims();
  index_t inner_size = 1;
  // default
  grid_dims = dim3(1, 1, 1);
  int inner_dim_exclude = 0;
  if (nd == 3)
  {
    grid_dims.y = region[-1];
    ++inner_dim_exclude;
  }
  else if (nd > 3)
  {
    grid_dims.y = region[-2];
    grid_dims.z = region[-1];
    inner_dim_exclude += 2;
  }
  for (int i = 0; i < nd - inner_dim_exclude; ++i)
  {
    inner_size *= region[i];
  }
  thread_work_size = std::min(
    (int) ((inner_size + BLOCK_SIZE - 1) / BLOCK_SIZE), MAX_THREAD_WORK_SIZE);
  int const block_work_size = BLOCK_SIZE * thread_work_size;
  size_t const num_blocks_per_space =
    (inner_size + block_work_size - 1) / block_work_size;
  grid_dims.x = num_blocks_per_space;
  return;
}

template <int BLOCK_SIZE, int MAX_THREAD_WORK_SIZE>
void get_grid_dims2(Shape const& region,
                    dim3& grid_dims,
                    int& thread_work_size,
                    int& inner_dim,
                    int& num_inner_blocks)
{
  int const nd = region.num_dims();
  // default
  grid_dims = dim3(1, 1, 1);

  // determine which dimensions are traversed by a single block
  index_t inner_size = 1;
  inner_dim = 0;
  for (; inner_dim < nd; ++inner_dim)
  {
    inner_size *= region[inner_dim];
    if (inner_size >= BLOCK_SIZE * MAX_THREAD_WORK_SIZE)
    {
      break;
    }
  }
  if (inner_dim == nd)
  {
    inner_dim = nd - 1;
  }

  thread_work_size = std::min(
    (int) ((inner_size + BLOCK_SIZE - 1) / BLOCK_SIZE), MAX_THREAD_WORK_SIZE);
  int const block_work_size = BLOCK_SIZE * thread_work_size;
  num_inner_blocks = (inner_size + block_work_size - 1) / block_work_size;

  int num_outer_blocks = 1;
  for (int i = inner_dim + 1; i < nd; ++i)
  {
    num_outer_blocks *= region[i];
  }

  grid_dims.x = num_inner_blocks * num_outer_blocks;
  return;
}

}  // namespace algorithms_cuda
}  // namespace tensor
}  // namespace distconv
