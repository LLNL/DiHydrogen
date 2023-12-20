////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/fixed_size_tuple.hpp"

// Test the constexpr version of FixedSizeTuple.

using TestFixedSizeTuple = h2::FixedSizeTuple<int, std::size_t, 4>;

// Empty tuple.
namespace empty_test {
constexpr TestFixedSizeTuple test_tuple;
static_assert(test_tuple.size() == 0,
              "Empty tuple does not have size 0");

constexpr TestFixedSizeTuple test_tuple2;
static_assert(test_tuple == test_tuple2,
              "Equality comparison for empty tuple is wrong");

static_assert(h2::product<int>(test_tuple) == 1,
              "Product of empty tuple is wrong.");
static_assert(h2::inner_product<int>(test_tuple, test_tuple2) == 0,
              "Inner product of empty tuples is wrong");

constexpr auto test_tuple_copy = test_tuple;
static_assert(test_tuple_copy.size() == 0,
              "Empty tuple copy does not have size 0");

constexpr auto test_tuple_copy_construct(test_tuple);
static_assert(test_tuple_copy_construct.size() == 0,
              "Copy-constructed empty tuple does not have size 0");

constexpr auto test_tuple_move = std::move(test_tuple);
static_assert(test_tuple_move.size() == 0,
              "Empty tuple move does not have size 0");

constexpr auto test_tuple_move_construct(std::move(test_tuple_copy));
static_assert(test_tuple_move_construct.size() == 0,
              "Move-constructed empty tuple does not have size 0");
}

// Sized tuple.
namespace sized_test {
constexpr TestFixedSizeTuple test_tuple(1, 2);
static_assert(test_tuple.size() == 2,
              "Sized tuple has wrong size");
static_assert(test_tuple[0] == 1,
              "Sized tuple index 0 has wrong value");
static_assert(test_tuple[1] == 2,
              "Sized tuple index 1 has wrong value");
constexpr TestFixedSizeTuple test_tuple2(1, 2);
static_assert(test_tuple == test_tuple2,
              "Sized tuple comparison is wrong");
constexpr TestFixedSizeTuple test_tuple3(2, 1);
static_assert(test_tuple != test_tuple3,
              "Sized tuple comparison is wrong");

static_assert(h2::product<int>(test_tuple) == 2,
              "Sized tuple product has wrong value");
static_assert(h2::inner_product<int>(test_tuple, test_tuple2) == 5,
              "Inner product of sized tuples has wrong value");

constexpr auto test_tuple_copy = test_tuple;
static_assert(test_tuple_copy.size() == 2,
              "Sized tuple copy has wrong size");
static_assert(test_tuple_copy[0] == 1,
              "Sized tuple copy index 0 has wrong value");
static_assert(test_tuple_copy[1] == 2,
              "Sized tuple copy index 1 has wrong value");

constexpr auto test_tuple_copy_construct(test_tuple);
static_assert(test_tuple_copy_construct.size() == 2,
              "Copy-constructed sized tuple has wrong size");
static_assert(test_tuple_copy_construct[0] == 1,
              "Copy-constructed sized tuple index 0 has wrong value");
static_assert(test_tuple_copy_construct[1] == 2,
              "Copy-constructed sized tuple index 1 has wrong value");

constexpr auto test_tuple_move = std::move(test_tuple);
static_assert(test_tuple_move.size() == 2,
              "Moved sized tuple has wrong size");
static_assert(test_tuple_move[0] == 1,
              "Moved sized tuple index 0 has wrong value");
static_assert(test_tuple_move[1] == 2,
              "Moved sized tuple index 1 has wrong value");

constexpr TestFixedSizeTuple test_tuple_move_construct(std::move(test_tuple_copy));
static_assert(test_tuple_move_construct.size() == 2,
              "Move-constructed sized tuple has wrong size");
static_assert(test_tuple_move_construct[0] == 1,
              "Move-constructed sized tuple index 0 has wrong value");
static_assert(test_tuple_move_construct[1] == 2,
              "Moved-constructed sized tuple index 1 has wrong value");
}

// Tuple padding.
namespace padding_test {
constexpr TestFixedSizeTuple test_empty_tuple_pad(h2::TuplePad<TestFixedSizeTuple>(0));
static_assert(test_empty_tuple_pad.size() == 0,
              "Padded empty tuple does not have size 0");

constexpr TestFixedSizeTuple test_empty_tuple_pad2(h2::TuplePad<TestFixedSizeTuple>(0, 1));
static_assert(test_empty_tuple_pad.size() == 0,
              "Padded empty tuple does not have size 0");

constexpr TestFixedSizeTuple test_sized_tuple_pad1(h2::TuplePad<TestFixedSizeTuple>(2));
static_assert(test_sized_tuple_pad1.size() == 2,
              "Padded sized tuple has wrong size");
static_assert(test_sized_tuple_pad1[0] == 0,
              "Padded sized tuple index 0 has wrong value");
static_assert(test_sized_tuple_pad1[1] == 0,
              "Padded sized tuple index 1 has wrong value");

constexpr TestFixedSizeTuple test_sized_tuple_pad2(h2::TuplePad<TestFixedSizeTuple>(2), 1, 2);
static_assert(test_sized_tuple_pad2.size() == 2,
              "Padded sized tuple has wrong size");
static_assert(test_sized_tuple_pad2[0] == 1,
              "Padded sized tuple index 0 has wrong value");
static_assert(test_sized_tuple_pad2[1] == 2,
              "Padded sized tuple index 1 has wrong value");

constexpr TestFixedSizeTuple test_sized_tuple_pad3(h2::TuplePad<TestFixedSizeTuple>(4), 1, 2);
static_assert(test_sized_tuple_pad3.size() == 4,
              "Padded sized tuple has wrong size");
static_assert(test_sized_tuple_pad3[0] == 1,
              "Padded sized tuple index 0 has wrong value");
static_assert(test_sized_tuple_pad3[1] == 2,
              "Padded sized tuple index 1 has wrong value");
static_assert(test_sized_tuple_pad3[2] == 0,
              "Padded sized tuple index 2 has wrong value");
static_assert(test_sized_tuple_pad3[3] == 0,
              "Padded sized tuple index 3 has wrong value");
}

namespace last_test
{

constexpr TestFixedSizeTuple test_last{1, 2, 3};
static_assert(last(test_last) == 3, "last(tuple) gives wrong value");

} // namespace last_test

namespace init_test
{

constexpr TestFixedSizeTuple test_init{3, 2, 1};
static_assert(init(test_init) == TestFixedSizeTuple{3, 2},
              "init(tuple) gives wrong value");

} // namespace init_test

// Sized tuple.
