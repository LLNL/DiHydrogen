// @H2_LICENSE_TEXT@

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/HaskellAccessors.hpp"
#include "h2/meta/typelist/LispAccessors.hpp"

using namespace h2::meta;
using namespace h2::meta::tlist;

// Testing Cons
static_assert(EqV<Cons<int,Empty>,TL<int>>(), "Cons to empty.");
static_assert(!EqV<Cons<int,Empty>,TL<float>>(), "Cons to empty.");
static_assert(EqV<Cons<int,TL<float>>,TL<int,float>>(), "Cons to nonempty.");

// Testing Prepend
static_assert(EqV<Prepend<int,Empty>,TL<int>>(), "Prepend to empty.");
static_assert(!EqV<Prepend<int,Empty>,TL<float>>(), "Prepend to empty.");
static_assert(EqV<Prepend<int,TL<float>>,TL<int,float>>(),
              "Prepend to nonempty.");

// Testing Car
static_assert(EqV<Car<Empty>, Nil>(),
              "Car of empty list is empty.");
static_assert(EqV<Car<TL<float>>,float>(),
              "Car of length one list");
static_assert(EqV<Car<TL<int, float>>,int>(),
              "Car of length two list");
static_assert(!EqV<Car<TL<int, float>>,float>(),
              "Car of length two list");

// Testing Head
static_assert(EqV<Head<Empty>, Nil>(),
              "Head of empty list is empty.");
static_assert(EqV<Head<TL<float>>,float>(),
              "Head of length one list");
static_assert(EqV<Head<TL<int, float>>,int>(),
              "Head of length two list");
static_assert(!EqV<Head<TL<int, float>>,float>(),
              "Head of length two list");

// Testing Cdr
static_assert(EqV<Cdr<Empty>, Empty>(), "Cdr of empty list is empty.");
static_assert(EqV<Cdr<TL<float>>,Empty>(), "Cdr of length one list");
static_assert(EqV<Cdr<TL<int, float>>,TL<float>>(),
              "Cdr of length two list");
static_assert(!EqV<Cdr<TL<int, float>>,float>(),
              "Car of length two list");

// Testing Tail
static_assert(EqV<Tail<Empty>, Empty>(), "Tail of empty list is empty.");
static_assert(EqV<Tail<TL<float>>,Empty>(), "Tail of length one list");
static_assert(EqV<Tail<TL<int, float>>,TL<float>>(),
              "Tail of length two list");
static_assert(!EqV<Tail<TL<int, float>>,float>(),
              "Car of length two list");

// Uniquely Haskell-y stuff

// Testing Last
static_assert(EqV<Last<Empty>, Nil>(), "Last of an empty list is nil.");
static_assert(EqV<Last<TL<float>>, float>(), "Last of an single-entry list.");
static_assert(EqV<Last<TL<int, char, float>>, float>(),
              "Last of a multi-entry list.");

// Testing Init
static_assert(EqV<Init<Empty>, Empty>(), "Init of an empty list is empty.");
static_assert(EqV<Init<TL<float>>, Empty>(),
              "Init of a single-entry list is empty.");
static_assert(EqV<Init<TL<int, char, float>>, TL<int, char>>(),
              "Init of a multi-entry list.");

// Quick sanity check
namespace static_test_accessor_sanity
{
using TList = TL<bool, char, short, int, long>;
static_assert(EqV<Cons<Car<TList>, Cdr<TList>>, TList>(),
              "Consing the car to the cdr gives the original list back.");
} // namespace static_test_accessor_sanity

// Testing the combinatorial craziness that is the CL c[ad]*r stuff.

namespace static_test_accessor_crazy
{
using TList1 = TL<bool, char, short>;
using TList2 = TL<TList1, short, int>;
using TList3 = TL<TList2, TList1, int, long>;
using TList4 = TL<TList3, TList2, TList1, int, long>;
using TList = TL<TList4, TList3, TList2, long, long long>;

static_assert(EqV<Car<TList>, TList4>(), "Car (again)");
static_assert(EqV<Cdr<TList>, TL<TList3, TList2, long, long long>>(),
              "Cdr (again)");

static_assert(EqV<Caar<TList>, TList3>(), "Car of car");
static_assert(EqV<Cadr<TList>, TList3>(), "Car of cdr");
static_assert(EqV<Cdar<TList>, TL<TList2, TList1, int, long>>(), "Cdr of car");
static_assert(EqV<Cddr<TList>, TL<TList2, long, long long>>(), "Cdr of car");

static_assert(EqV<Caaar<TList>, TList2>(), "Car of car of car");
static_assert(EqV<Caadr<TList>, TList2>(), "Car of car of cdr");
static_assert(EqV<Cadar<TList>, TList2>(), "Car of cdr of car");
static_assert(EqV<Caddr<TList>, TList2>(), "Car of cdr of cdr");
static_assert(EqV<Cdaar<TList>, TL<TList1, int, long>>(),
              "Cdr of car of car");
static_assert(EqV<Cdadr<TList>, TL<TList1, int, long>>(),
              "Cdr of car of cdr");
static_assert(EqV<Cddar<TList>, TL<TList1, int, long>>(),
              "Cdr of cdr of car");
static_assert(EqV<Cdddr<TList>, TL<long, long long>>(),
              "Cdr of cdr of cdr");

static_assert(EqV<Caaaar<TList>, TList1>(), "Car of car of car of car");
static_assert(EqV<Caaadr<TList>, TList1>(), "Car of car of car of cdr");
static_assert(EqV<Caadar<TList>, TList1>(), "Car of car of cdr of car");
static_assert(EqV<Caaddr<TList>, TList1>(), "Car of car of cdr of cdr");
static_assert(EqV<Cadaar<TList>, TList1>(), "Car of cdr of car of car");
static_assert(EqV<Cadadr<TList>, TList1>(), "Car of cdr of car of cdr");
static_assert(EqV<Caddar<TList>, TList1>(), "Car of cdr of cdr of car");
static_assert(EqV<Cadddr<TList>, long>(), "Car of cdr of cdr of cdr");
static_assert(EqV<Cdaaar<TList>, TL<short, int>>(),
              "Cdr of car of car of car");
static_assert(EqV<Cdaadr<TList>, TL<short, int>>(),
              "Cdr of car of car of cdr");
static_assert(EqV<Cdadar<TList>, TL<short, int>>(),
              "Cdr of car of cdr of car");
static_assert(EqV<Cdaddr<TList>, TL<short, int>>(),
              "Cdr of car of cdr of cdr");
static_assert(EqV<Cddaar<TList>, TL<int, long>>(),
              "Cdr of cdr of car of car");
static_assert(EqV<Cddadr<TList>, TL<int, long>>(),
              "Cdr of cdr of car of cdr");
static_assert(EqV<Cdddar<TList>, TL<int, long>>(),
              "Cdr of cdr of cdr of car");
static_assert(EqV<Cddddr<TList>, TL<long long>>(),
              "Cdr of cdr of cdr of cdr");
}// namespace static_test_accessor_crazy
