////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <unordered_map>

#include "h2/tensor/fixed_size_tuple.hpp"
#include "h2/tensor/tuple_utils.hpp"

// Test the constexpr version of FixedSizeTuple.

using TestFixedSizeTuple = h2::FixedSizeTuple<int, std::size_t, 4>;

// Empty tuple.
namespace empty_test {
constexpr TestFixedSizeTuple test_tuple;
static_assert(test_tuple.size() == 0);
static_assert(test_tuple.is_empty());

constexpr auto converted_tuple =
    h2::FixedSizeTuple<long long int, std::size_t, 4>::convert_from(test_tuple);
static_assert(converted_tuple.size() == 0);
static_assert(converted_tuple.is_empty());

constexpr TestFixedSizeTuple test_tuple2;
static_assert(test_tuple == test_tuple2);

static_assert(h2::product<int>(test_tuple) == 1);
static_assert(h2::inner_product<int>(test_tuple, test_tuple2) == 0);
static_assert(h2::prefix_product<int>(test_tuple) == TestFixedSizeTuple{});
static_assert(!h2::any_of(test_tuple,
                          [](TestFixedSizeTuple::type x) { return x == 0; }));
static_assert(h2::all_of(test_tuple,
                         [](TestFixedSizeTuple::type x) { return x == 0; }));
static_assert(h2::map(test_tuple, [](TestFixedSizeTuple::type x) { return x; })
              == TestFixedSizeTuple{});
static_assert(h2::map_index(test_tuple,
                            [](TestFixedSizeTuple::size_type x) {
                              return test_tuple[x];
                            })
              == TestFixedSizeTuple{});
static_assert(h2::filter(test_tuple,
                         [](TestFixedSizeTuple::type x) { return x == 0; })
              == TestFixedSizeTuple{});
static_assert(h2::filter_index(test_tuple,
                               [](TestFixedSizeTuple::size_type x) {
                                 return x == 0;
                               })
              == TestFixedSizeTuple{});

static_assert(test_tuple.begin() == test_tuple.end());
static_assert(test_tuple.cbegin() == test_tuple.cend());
static_assert(test_tuple.rbegin() == test_tuple.rend());
static_assert(test_tuple.crbegin() == test_tuple.crend());

constexpr auto test_tuple_copy = test_tuple;
static_assert(test_tuple_copy.size() == 0);

constexpr auto test_tuple_copy_construct(test_tuple);
static_assert(test_tuple_copy_construct.size() == 0);

constexpr auto test_tuple_move = std::move(test_tuple);
static_assert(test_tuple_move.size() == 0);

constexpr auto test_tuple_move_construct(std::move(test_tuple_copy));
static_assert(test_tuple_move_construct.size() == 0);

}  // namespace empty_test

// Sized tuple.
namespace sized_test {
constexpr TestFixedSizeTuple test_tuple(1, 2);
static_assert(test_tuple.size() == 2);
static_assert(test_tuple[0] == 1);
static_assert(test_tuple[1] == 2);
static_assert(!test_tuple.is_empty());
constexpr TestFixedSizeTuple test_tuple2(1, 2);
static_assert(test_tuple == test_tuple2);
constexpr TestFixedSizeTuple test_tuple3(2, 1);
static_assert(test_tuple != test_tuple3);

constexpr auto converted_tuple =
    h2::FixedSizeTuple<long long int, std::size_t, 4>::convert_from(test_tuple);
static_assert(converted_tuple.size() == 2);
static_assert(converted_tuple[0] == 1);
static_assert(converted_tuple[1] == 2);
static_assert(!converted_tuple.is_empty());
static_assert(converted_tuple == test_tuple);

static_assert(h2::product<int>(test_tuple) == 2);
static_assert(h2::inner_product<int>(test_tuple, test_tuple2) == 5);
static_assert(h2::prefix_product<int>(test_tuple) == TestFixedSizeTuple(1, 1));
static_assert(h2::prefix_product<int>(TestFixedSizeTuple(1, 2, 3))
              == TestFixedSizeTuple(1, 1, 2));
static_assert(h2::any_of(test_tuple,
                         [](TestFixedSizeTuple::type x) { return x == 1; }));
static_assert(!h2::any_of(test_tuple,
                          [](TestFixedSizeTuple::type x) { return x == 3; }));
static_assert(h2::all_of(test_tuple,
                         [](TestFixedSizeTuple::type x) {
                           return x == 1 || x == 2;
                         }));
static_assert(!h2::all_of(test_tuple,
                          [](TestFixedSizeTuple::type x) {
                            return x == 1 || x == 3;
                          }));
static_assert(h2::map(test_tuple,
                      [](TestFixedSizeTuple::type x) { return x + 1; })
              == TestFixedSizeTuple{2, 3});
static_assert(h2::map(test_tuple,
                            [](TestFixedSizeTuple::type x) { return x == 1; })
                  == h2::FixedSizeTuple<bool,
                                        typename TestFixedSizeTuple::size_type,
                                        TestFixedSizeTuple::max_size>(true,
                                                                      false));
static_assert(h2::map_index(test_tuple,
                            [](TestFixedSizeTuple::size_type x) {
                              return test_tuple[x] + 1;
                            })
              == TestFixedSizeTuple{2, 3});
static_assert(h2::map_index(test_tuple,
                                  [](TestFixedSizeTuple::size_type x) {
                                    return test_tuple[x] == 1;
                                  })
                  == h2::FixedSizeTuple<bool,
                                        typename TestFixedSizeTuple::size_type,
                                        TestFixedSizeTuple::max_size>(true,
                                                                      false));
static_assert(h2::filter(test_tuple,
                         [](TestFixedSizeTuple::type x) { return x == 1; })
              == TestFixedSizeTuple{1});
static_assert(h2::filter_index(test_tuple,
                               [](TestFixedSizeTuple::size_type x) {
                                 return x == 0;
                               })
              == TestFixedSizeTuple{1});

static_assert(*test_tuple.begin() == 1);
static_assert(*std::next(test_tuple.begin()) == 2);
static_assert(test_tuple.begin() + test_tuple.size() == test_tuple.end());

static_assert(*test_tuple.rbegin() == 2);
static_assert(*std::next(test_tuple.rbegin()) == 1);
static_assert(test_tuple.rbegin() + test_tuple.size() == test_tuple.rend());

static_assert(test_tuple.front() == 1);
static_assert(test_tuple.back() == 2);

constexpr auto test_tuple_copy = test_tuple;
static_assert(test_tuple_copy.size() == 2);
static_assert(test_tuple_copy[0] == 1);
static_assert(test_tuple_copy[1] == 2);

constexpr auto test_tuple_copy_construct(test_tuple);
static_assert(test_tuple_copy_construct.size() == 2);
static_assert(test_tuple_copy_construct[0] == 1);
static_assert(test_tuple_copy_construct[1] == 2);

constexpr auto test_tuple_move = std::move(test_tuple);
static_assert(test_tuple_move.size() == 2);
static_assert(test_tuple_move[0] == 1);
static_assert(test_tuple_move[1] == 2);

constexpr TestFixedSizeTuple test_tuple_move_construct(std::move(test_tuple_copy));
static_assert(test_tuple_move_construct.size() == 2);
static_assert(test_tuple_move_construct[0] == 1);
static_assert(test_tuple_move_construct[1] == 2);
}  // namespace sized_test

// Tuple padding.
namespace padding_test {
constexpr TestFixedSizeTuple test_empty_tuple_pad(h2::TuplePad<TestFixedSizeTuple>(0));
static_assert(test_empty_tuple_pad.size() == 0);

constexpr TestFixedSizeTuple test_empty_tuple_pad2(h2::TuplePad<TestFixedSizeTuple>(0, 1));
static_assert(test_empty_tuple_pad.size() == 0);

constexpr TestFixedSizeTuple test_sized_tuple_pad1(h2::TuplePad<TestFixedSizeTuple>(2));
static_assert(test_sized_tuple_pad1.size() == 2);
static_assert(test_sized_tuple_pad1[0] == 0);
static_assert(test_sized_tuple_pad1[1] == 0);

constexpr TestFixedSizeTuple test_sized_tuple_pad2(h2::TuplePad<TestFixedSizeTuple>(2), 1, 2);
static_assert(test_sized_tuple_pad2.size() == 2);
static_assert(test_sized_tuple_pad2[0] == 1);
static_assert(test_sized_tuple_pad2[1] == 2);

constexpr TestFixedSizeTuple test_sized_tuple_pad3(h2::TuplePad<TestFixedSizeTuple>(4), 1, 2);
static_assert(test_sized_tuple_pad3.size() == 4);
static_assert(test_sized_tuple_pad3[0] == 1);
static_assert(test_sized_tuple_pad3[1] == 2);
static_assert(test_sized_tuple_pad3[2] == 0);
static_assert(test_sized_tuple_pad3[3] == 0);
}

namespace init_test
{

constexpr TestFixedSizeTuple test_init{3, 2, 1};
static_assert(init(test_init) == TestFixedSizeTuple{3, 2});

}  // namespace init_test

namespace init_n_test
{
constexpr TestFixedSizeTuple test(1, 2, 3);
static_assert(h2::init_n(test, std::size_t{0}) == TestFixedSizeTuple{});
static_assert(h2::init_n(test, std::size_t{1}) == TestFixedSizeTuple(1));
static_assert(h2::init_n(test, std::size_t{3}) == TestFixedSizeTuple(1, 2, 3));
constexpr TestFixedSizeTuple test_empty;
static_assert(h2::init_n(test_empty, std::size_t{0}) == TestFixedSizeTuple{});
}  // namespace init_n_test

// Ensure the hash specialization is picked up.
std::unordered_map<TestFixedSizeTuple, int> hash_test_map;

// Sized tuple.
