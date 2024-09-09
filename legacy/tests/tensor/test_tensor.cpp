#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_process.hpp"
#include "distconv/util/util.hpp"
#include "test_tensor.hpp"

using namespace distconv::tensor;

template <typename Allocator>
int test_view() {
  util::PrintStreamInfo() << "test_view";

  constexpr int ND = 2;
  constexpr int len = 4;
  Array<ND> shape = {len, len};

  int *raw_buffer = new int[len * len];
  for (int i = 0; i < len * len; ++i) {
    raw_buffer[i] = i;
  }

  using TensorType = Tensor<int, LocaleProcess, Allocator>;
  auto loc = get_locale<typename TensorType::locale_type>();
  auto dist = Distribution::make_localized_distribution(ND);
  TensorType t_view = get_tensor<TensorType>(shape, loc, dist);

  View(t_view, (const int*)raw_buffer);
  util::PrintStreamInfo() << "const view created";

  auto local_shape = t_view.get_local_shape();
  //auto base_offset = t_view.get_local_offset();
  const int *view_buf = t_view.get_const_buffer();
  for (index_t i = 0; i < local_shape[1]; ++i) {
    for (index_t j = 0; j < local_shape[0]; ++j) {
      IndexVector idx({j, i});
      int ref = get_linearlized_offset(t_view.get_global_index(idx),
                                       t_view.get_shape());
      int stored = view_buf[t_view.get_local_offset(idx)];
      if (ref != stored) {
        util::PrintStreamError()
            << "Mismatch at: " << idx
            << ", ref: " << ref << ", stored: " << stored;
        return -1;
      }
    }
  }
  return 0;
}

/*
  Usage: ./test_tensor
 */
int main(int argc, char *argv[]) {
  constexpr int ND = 3;
  using DataType = int;
  using LocaleType = LocaleProcess;
  using TensorType = Tensor<DataType, LocaleType, BaseAllocator>;

  auto dist = Distribution::make_localized_distribution(ND);
  assert0(test_alloc<TensorType>(Shape({1, 2, 3}), dist));
  assert0(test_data_access<TensorType>(Shape({2, 2, 2}), dist));

  using PitchedTensorType = Tensor<DataType, LocaleType,
                                   BasePitchedAllocator<4>>;
  assert0(test_alloc<PitchedTensorType>(Shape({2, 2, 2}), dist));
  assert0(test_data_access<PitchedTensorType>(Shape({2, 2, 2}), dist));

  assert0(test_view<BaseAllocator>());

  util::PrintStreamInfo() << "Completed successfully.";
  return 0;
}
