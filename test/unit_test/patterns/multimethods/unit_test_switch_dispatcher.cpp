// @H2_LICENSE_TEXT@

#include <catch2/catch.hpp>

#include "h2/meta/TypeList.hpp"
#include "h2/patterns/multimethods/SwitchDispatcher.hpp"

using namespace h2::meta;
using namespace h2::multimethods;

namespace
{
struct base
{
    virtual ~base() = default;
};
struct derived_one : base
{
    static constexpr unsigned value = 1;
};
struct derived_two : base
{
    static constexpr unsigned value = 8;
};
struct derived_three : base
{
    static constexpr unsigned value = 64;
};

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

    template <typename... Ts>
    int DeductionError(Ts&&...)
    {
        throw std::logic_error("Failed to deduce the type of an argument");
    }

    template <typename... Ts>
    int DispatchError(Ts&&...)
    {
        throw std::logic_error("No viable overload found.");
    }

    template <typename T1, typename T2, typename T3>
    int operator()(T1 const&, T2 const&, T3 const&)
    {
        return static_cast<int>(First<T1>() + Second<T2>() + Third<T3>());
    }
};

struct TestFunctorWithArgs
{
    int operator()(int x, derived_one const&, derived_one const&)
    {
        return 0 + x;
    }
    int operator()(int x, derived_two const&, derived_one const&)
    {
        return 1 + x;
    }
    int operator()(int x, derived_one const&, derived_two const&)
    {
        return 2 + x;
    }
    int operator()(int x, derived_two const&, derived_two const&)
    {
        return 3 + x;
    }

    template <typename... Ts>
    int DeductionError(Ts&&...)
    {
        throw std::logic_error("Failed to deduce the type of an argument");
    }

    template <typename... Ts>
    int DispatchError(Ts&&...)
    {
        throw std::logic_error("No viable overload found.");
    }
};

} // namespace

using DTypes = TL<derived_one, derived_two, derived_three>;
using DTypesNoD3 = TL<derived_one, derived_two>;

TEST_CASE("Switch dispatcher", "[h2][utils][multimethods]")
{
    derived_one d1;
    derived_two d2;
    derived_three d3;

    base* d1_b = &d1;
    base* d2_b = &d2;
    base* d3_b = &d3;

    SECTION("Double dispatch, basic functor with all deduced arguments.")
    {
        using Dispatcher =
            SwitchDispatcher<TestFunctor, int, base, DTypes, base, DTypes>;

        TestFunctor f;
        CHECK(Dispatcher::Exec(f, *d1_b, *d1_b) == f(d1, d1));
        CHECK(Dispatcher::Exec(f, *d1_b, *d2_b) == f(d1, d2));
        CHECK(Dispatcher::Exec(f, *d2_b, *d1_b) == f(d2, d1));
        CHECK(Dispatcher::Exec(f, *d2_b, *d2_b) == f(d2, d2));

        CHECK_THROWS_AS(Dispatcher::Exec(f, *d3_b, *d1_b), std::logic_error);
        CHECK_THROWS_AS(Dispatcher::Exec(f, *d3_b, *d2_b), std::logic_error);
        CHECK_THROWS_AS(Dispatcher::Exec(f, *d1_b, *d3_b), std::logic_error);
        CHECK_THROWS_AS(Dispatcher::Exec(f, *d2_b, *d3_b), std::logic_error);
    }

    SECTION("Triple dispatch")
    {
        using Dispatcher = SwitchDispatcher<
            TestFunctor, int, base, DTypes, base, DTypes, base, DTypesNoD3>;
        TestFunctor f;
        CHECK(Dispatcher::Exec(f, *d1_b, *d1_b, *d1_b) == f(d1, d1, d1));
        CHECK(Dispatcher::Exec(f, *d1_b, *d1_b, *d2_b) == f(d1, d1, d2));
        CHECK(Dispatcher::Exec(f, *d1_b, *d2_b, *d1_b) == f(d1, d2, d1));
        CHECK(Dispatcher::Exec(f, *d1_b, *d2_b, *d2_b) == f(d1, d2, d2));
        CHECK(Dispatcher::Exec(f, *d2_b, *d1_b, *d1_b) == f(d2, d1, d1));
        CHECK(Dispatcher::Exec(f, *d2_b, *d1_b, *d2_b) == f(d2, d1, d2));
        CHECK(Dispatcher::Exec(f, *d2_b, *d2_b, *d1_b) == f(d2, d2, d1));
        CHECK(Dispatcher::Exec(f, *d2_b, *d2_b, *d2_b) == f(d2, d2, d2));

        CHECK_THROWS_AS(
            Dispatcher::Exec(f, *d1_b, *d1_b, *d3_b), std::logic_error);
        CHECK_THROWS_AS(
            Dispatcher::Exec(f, *d1_b, *d2_b, *d3_b), std::logic_error);
        CHECK_THROWS_AS(
            Dispatcher::Exec(f, *d1_b, *d3_b, *d3_b), std::logic_error);
        CHECK_THROWS_AS(
            Dispatcher::Exec(f, *d2_b, *d1_b, *d3_b), std::logic_error);
        CHECK_THROWS_AS(
            Dispatcher::Exec(f, *d2_b, *d2_b, *d3_b), std::logic_error);
        CHECK_THROWS_AS(
            Dispatcher::Exec(f, *d2_b, *d3_b, *d3_b), std::logic_error);
        CHECK_THROWS_AS(
            Dispatcher::Exec(f, *d3_b, *d1_b, *d3_b), std::logic_error);
        CHECK_THROWS_AS(
            Dispatcher::Exec(f, *d3_b, *d2_b, *d3_b), std::logic_error);
        CHECK_THROWS_AS(
            Dispatcher::Exec(f, *d3_b, *d3_b, *d3_b), std::logic_error);
    }

    SECTION("Functor with additional arguments.")
    {
        using Dispatcher = SwitchDispatcher<
            TestFunctorWithArgs, int, base, DTypes, base, DTypes>;

        TestFunctorWithArgs f;
        CHECK(Dispatcher::Exec(f, *d1_b, *d1_b, 13) == f(13, d1, d1));
        CHECK(Dispatcher::Exec(f, *d1_b, *d2_b, 13) == f(13, d1, d2));
        CHECK(Dispatcher::Exec(f, *d2_b, *d1_b, 13) == f(13, d2, d1));
        CHECK(Dispatcher::Exec(f, *d2_b, *d2_b, 13) == f(13, d2, d2));
    }
}
