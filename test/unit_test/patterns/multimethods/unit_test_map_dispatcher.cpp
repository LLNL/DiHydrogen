////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/TypeList.hpp"
#include "h2/patterns/multimethods/MapDispatcher.hpp"

#include <catch2/catch_template_test_macros.hpp>

using namespace h2::meta;
using namespace h2::multimethods;

namespace
{
struct base
{
  virtual ~base() = default;
  virtual int val() const = 0;
};
struct derived_one final : base
{
  static constexpr unsigned value = 1;
  int val() const final { return value; }
};
struct derived_two final : base
{
  static constexpr unsigned value = 8;
  int val() const final { return value; }
};
struct derived_thr final : base
{
  static constexpr unsigned value = 64;
  int val() const final { return value; }
};
struct derived_fou final : base
{
  int val() const final { return 512; }
};

struct otherbase
{
  virtual ~otherbase() = default;
};

struct other_d1 : otherbase
{};

struct other_d2 : otherbase
{};

template <typename T>
constexpr unsigned First()
{
  return T::value;
}

template <typename T>
constexpr unsigned Second()
{
  return First<T>() << 1;
}

template <typename T>
constexpr unsigned Third()
{
  return Second<T>() << 1;
}

static_assert(Second<derived_one>() == 2, "");
static_assert(Third<derived_one>() == 4, "");
static_assert(Second<derived_two>() == 16, "");
static_assert(Third<derived_two>() == 32, "");

struct TestFunctor
{
  int operator()(derived_one const&, derived_one const&) { return 0; }
  int operator()(derived_two const&, derived_one const&) { return 1; }
  int operator()(derived_one const&, derived_two const&) { return 2; }
  int operator()(derived_two const&, derived_two const&) { return 3; }

  template <typename T1, typename T2, typename T3>
  int operator()(T1 const&, T2 const&, T3 const&)
  {
    return static_cast<int>(First<T1>() + Second<T2>() + Third<T3>());
  }
};

int test_function(derived_two const& x, derived_one const& y)
{
  return 52;
}

struct OtherInteractor
{
  int operator()(derived_one const&, other_d1 const&) { return 11; }
  int operator()(derived_two const&, other_d1 const&) { return 13; }
  int operator()(derived_one const&, other_d2 const&) { return 17; }
  int operator()(derived_two const&, other_d2 const&) { return 19; }
};

int other_test_function(derived_fou const&, other_d1 const&)
{
  return 23;
}

}  // namespace

// This is a silly hack to bind the CasterT into a real class.
template <template <class, class> class CasterT>
struct DispatcherMaker
{
  template <typename ReturnT, typename BaseTL>
  static auto get_dispatcher() -> MapDispatcher<ReturnT, BaseTL, CasterT>
  {
    return MapDispatcher<ReturnT, BaseTL, CasterT>{};
  }
};

using DispatcherMakers =
  TL<DispatcherMaker<DynamicDownCaster>, DispatcherMaker<StaticDownCaster>>;

TEMPLATE_LIST_TEST_CASE("Log-time map-based dispatcher",
                        "[utilities][multimethods][mapdispatcher]",
                        DispatcherMakers)
{
  using DispatcherMakerT = TestType;

  derived_one d1;
  derived_two d2;
  derived_thr d3;
  derived_fou d4;

  base& d1_b = d1;
  base& d2_b = d2;
  base& d3_b = d3;
  base& d4_b = d4;

  other_d1 od1;
  other_d2 od2;

  otherbase& od1_b = od1;
  otherbase& od2_b = od2;

  SECTION("Double dispatch with a basic functor.")
  {
    auto d = DispatcherMakerT::
      template get_dispatcher<int, tlist::Repeat<base const, 2UL>>();
    TestFunctor f;

    REQUIRE_NOTHROW(d.template add<derived_one, derived_one>(f));
    REQUIRE_NOTHROW(d.template add<derived_one, derived_two>(f));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_one>(f));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_two>(f));

    CHECK(d.call(d1_b, d1_b) == f(d1, d1));
    CHECK(d.call(d2_b, d1_b) == f(d2, d1));
    CHECK(d.call(d1_b, d2_b) == f(d1, d2));
    CHECK(d.call(d2_b, d2_b) == f(d2, d2));

    CHECK_THROWS_AS(d.call(d3_b, d1_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d3_b, d2_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d1_b, d3_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d2_b, d3_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d3_b, d3_b), NoDispatchAdded);
  }

  SECTION("Double dispatch with a basic functor, different bases.")
  {
    auto d = DispatcherMakerT::
      template get_dispatcher<int, TL<base const, otherbase const>>();
    OtherInteractor f;

    REQUIRE_NOTHROW(H2_MDISP_ADD(d, f, derived_one, other_d1));
    REQUIRE_NOTHROW(d.template add<derived_two, other_d1>(f));
    REQUIRE_NOTHROW(d.template add<derived_one, other_d2>(f));
    REQUIRE_NOTHROW(d.template add<derived_two, other_d2>(f));

    CHECK(d.call(d1_b, od1_b) == f(d1, od1));
    CHECK(d.call(d2_b, od1_b) == f(d2, od1));
    CHECK(d.call(d1_b, od2_b) == f(d1, od2));
    CHECK(d.call(d2_b, od2_b) == f(d2, od2));

    CHECK_THROWS_AS(d.call(d3_b, od1_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d3_b, od2_b), NoDispatchAdded);

    REQUIRE_NOTHROW(d.template add<derived_fou, other_d1>(other_test_function));
    CHECK(d.call(d4_b, od1_b) == other_test_function(d4, od1));

    REQUIRE_NOTHROW(
      H2_MDISP_ADD_FP(d, other_test_function, derived_fou, other_d1));
    CHECK(d.call(d4_b, od1_b) == other_test_function(d4, od1));

    CHECK_THROWS_AS(d.call(d4_b, od2_b), NoDispatchAdded);
  }

  SECTION("Registering a raw function pointer")
  {
    auto d = DispatcherMakerT::
      template get_dispatcher<int, tlist::Repeat<base const, 2UL>>();

    REQUIRE_NOTHROW(
      H2_MDISP_ADD_FP(d, test_function, derived_two, derived_one));
    REQUIRE(d.call(d2_b, d1_b) == 52);
  }

  SECTION("Calling add() replaces any existing dispatch")
  {
    auto d = DispatcherMakerT::
      template get_dispatcher<int, tlist::Repeat<base const, 2UL>>();

    // Test the macro version...
    REQUIRE_NOTHROW(H2_MDISP_ADD(d,
                                 ([](auto const&, auto const&) { return 13; }),
                                 derived_one,
                                 derived_two));

    REQUIRE(d.call(d1_b, d2_b) == 13);

    // And the non-macro version...
    REQUIRE_NOTHROW(d.template add<derived_one, derived_two>(
      [](derived_one const&, derived_two const&) { return 42; }));

    REQUIRE(d.call(d1_b, d2_b) == 42);
  }

  SECTION("Double dispatch, basic functor wrapped in lambda")
  {
    auto d = DispatcherMakerT::
      template get_dispatcher<int, tlist::Repeat<base const, 2UL>>();
    TestFunctor f;

    REQUIRE_NOTHROW(d.template add<derived_one, derived_one>(
      [f](derived_one const& x, derived_one const& y) mutable {
        return f(x, y);
      }));
    REQUIRE_NOTHROW(d.template add<derived_one, derived_two>(
      [f](derived_one const& x, derived_two const& y) mutable {
        return f(x, y);
      }));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_one>(
      [f](derived_two const& x, derived_one const& y) mutable {
        return f(x, y);
      }));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_two>(
      [f](derived_two const& x, derived_two const& y) mutable {
        return f(x, y);
      }));

    CHECK(d.call(d1_b, d1_b) == f(d1, d1));
    CHECK(d.call(d2_b, d1_b) == f(d2, d1));
    CHECK(d.call(d1_b, d2_b) == f(d1, d2));
    CHECK(d.call(d2_b, d2_b) == f(d2, d2));

    CHECK_THROWS_AS(d.call(d3_b, d1_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d3_b, d2_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d1_b, d3_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d2_b, d3_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d3_b, d3_b), NoDispatchAdded);
  }

  SECTION("Triple dispatch")
  {
    auto d = DispatcherMakerT::
      template get_dispatcher<int, tlist::Repeat<base const, 3UL>>();
    TestFunctor f;
    REQUIRE_NOTHROW(d.template add<derived_one, derived_one, derived_one>(f));
    REQUIRE_NOTHROW(d.template add<derived_one, derived_one, derived_two>(f));
    REQUIRE_NOTHROW(d.template add<derived_one, derived_one, derived_thr>(f));
    REQUIRE_NOTHROW(d.template add<derived_one, derived_two, derived_one>(f));
    REQUIRE_NOTHROW(d.template add<derived_one, derived_two, derived_two>(f));
    REQUIRE_NOTHROW(d.template add<derived_one, derived_two, derived_thr>(f));
    REQUIRE_NOTHROW(d.template add<derived_one, derived_thr, derived_one>(f));
    REQUIRE_NOTHROW(d.template add<derived_one, derived_thr, derived_two>(f));
    REQUIRE_NOTHROW(d.template add<derived_one, derived_thr, derived_thr>(f));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_one, derived_one>(f));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_one, derived_two>(f));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_one, derived_thr>(f));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_two, derived_one>(f));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_two, derived_two>(f));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_two, derived_thr>(f));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_thr, derived_one>(f));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_thr, derived_two>(f));
    REQUIRE_NOTHROW(d.template add<derived_two, derived_thr, derived_thr>(f));
    REQUIRE_NOTHROW(d.template add<derived_thr, derived_one, derived_one>(f));
    REQUIRE_NOTHROW(d.template add<derived_thr, derived_one, derived_two>(f));
    REQUIRE_NOTHROW(d.template add<derived_thr, derived_one, derived_thr>(f));
    REQUIRE_NOTHROW(d.template add<derived_thr, derived_two, derived_one>(f));
    REQUIRE_NOTHROW(d.template add<derived_thr, derived_two, derived_two>(f));
    REQUIRE_NOTHROW(d.template add<derived_thr, derived_two, derived_thr>(f));
    REQUIRE_NOTHROW(d.template add<derived_thr, derived_thr, derived_one>(f));
    REQUIRE_NOTHROW(d.template add<derived_thr, derived_thr, derived_two>(f));
    REQUIRE_NOTHROW(d.template add<derived_thr, derived_thr, derived_thr>(f));

    CHECK(d.call(d1_b, d1_b, d1_b) == f(d1, d1, d1));
    CHECK(d.call(d1_b, d1_b, d2_b) == f(d1, d1, d2));
    CHECK(d.call(d1_b, d1_b, d3_b) == f(d1, d1, d3));
    CHECK(d.call(d1_b, d2_b, d1_b) == f(d1, d2, d1));
    CHECK(d.call(d1_b, d2_b, d2_b) == f(d1, d2, d2));
    CHECK(d.call(d1_b, d2_b, d3_b) == f(d1, d2, d3));
    CHECK(d.call(d1_b, d3_b, d1_b) == f(d1, d3, d1));
    CHECK(d.call(d1_b, d3_b, d2_b) == f(d1, d3, d2));
    CHECK(d.call(d1_b, d3_b, d3_b) == f(d1, d3, d3));
    CHECK(d.call(d2_b, d1_b, d1_b) == f(d2, d1, d1));
    CHECK(d.call(d2_b, d1_b, d2_b) == f(d2, d1, d2));
    CHECK(d.call(d2_b, d1_b, d3_b) == f(d2, d1, d3));
    CHECK(d.call(d2_b, d2_b, d1_b) == f(d2, d2, d1));
    CHECK(d.call(d2_b, d2_b, d2_b) == f(d2, d2, d2));
    CHECK(d.call(d2_b, d2_b, d3_b) == f(d2, d2, d3));
    CHECK(d.call(d2_b, d3_b, d1_b) == f(d2, d3, d1));
    CHECK(d.call(d2_b, d3_b, d2_b) == f(d2, d3, d2));
    CHECK(d.call(d2_b, d3_b, d3_b) == f(d2, d3, d3));
    CHECK(d.call(d3_b, d1_b, d1_b) == f(d3, d1, d1));
    CHECK(d.call(d3_b, d1_b, d2_b) == f(d3, d1, d2));
    CHECK(d.call(d3_b, d1_b, d3_b) == f(d3, d1, d3));
    CHECK(d.call(d3_b, d2_b, d1_b) == f(d3, d2, d1));
    CHECK(d.call(d3_b, d2_b, d2_b) == f(d3, d2, d2));
    CHECK(d.call(d3_b, d2_b, d3_b) == f(d3, d2, d3));
    CHECK(d.call(d3_b, d3_b, d1_b) == f(d3, d3, d1));
    CHECK(d.call(d3_b, d3_b, d2_b) == f(d3, d3, d2));
    CHECK(d.call(d3_b, d3_b, d3_b) == f(d3, d3, d3));

    CHECK_THROWS_AS(d.call(d1_b, d1_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d1_b, d2_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d1_b, d3_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d2_b, d1_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d2_b, d2_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d2_b, d3_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d3_b, d1_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d3_b, d2_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d3_b, d3_b, d4_b), NoDispatchAdded);

    CHECK_THROWS_AS(d.call(d1_b, d4_b, d1_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d1_b, d4_b, d2_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d1_b, d4_b, d3_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d2_b, d4_b, d1_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d2_b, d4_b, d2_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d2_b, d4_b, d3_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d3_b, d4_b, d1_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d3_b, d4_b, d2_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d3_b, d4_b, d3_b), NoDispatchAdded);

    CHECK_THROWS_AS(d.call(d4_b, d1_b, d1_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d1_b, d2_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d1_b, d3_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d2_b, d1_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d2_b, d2_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d2_b, d3_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d3_b, d1_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d3_b, d2_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d3_b, d3_b), NoDispatchAdded);

    CHECK_THROWS_AS(d.call(d1_b, d4_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d2_b, d4_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d3_b, d4_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d1_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d2_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d3_b, d4_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d4_b, d1_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d4_b, d2_b), NoDispatchAdded);
    CHECK_THROWS_AS(d.call(d4_b, d4_b, d3_b), NoDispatchAdded);

    CHECK_THROWS_AS(d.call(d4_b, d4_b, d4_b), NoDispatchAdded);
  }
}
