////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_PATTERNS_MULTIMETHODS_SWITCHDISPATCHER_HPP_
#define H2_PATTERNS_MULTIMETHODS_SWITCHDISPATCHER_HPP_

#include "h2/meta/Core.hpp"
#include "h2/meta/TypeList.hpp"

#include <utility>

namespace h2
{
namespace multimethods
{
/** @brief Dispatch a functor call based on the dynamic type of the arguments.
 *
 *  @tparam FunctorT The type of the functor to dispatch. It must
 *          implement `operator()`. All overloads must have the same
 *          return type.
 *  @tparam ReturnT The return type of all overloads of `operator()`.
 *  @tparam ArgumentTs The types of the arguments to the
 *          functor. Arguments that are part of a `BaseTypesPair` will
 *          undergo dynamic deduction.
 *
 *  @section switch-dispatch-intro Introduction
 *
 *  The problem of multiple dispatch is that, occasionally, objects
 *  need to interact at the public API level via references to their
 *  base class(es) but have implementations that vary based on the
 *  concrete (dynamic) types of the objects. Handling this dispatch
 *  manually is messy, prone to duplication, and difficult to
 *  maintain. This dispatcher implements one solution to this problem
 *  by deducing the dynamic type of certain types of arguments in a
 *  brute-force fashion. That is, code is generated for all possible
 *  combinations of dynamically-deduced types, though some may end in
 *  exceptions being thrown if no viable dispatch is found.
 *
 *  @section switch-dispatch-algo Algorithm
 *
 *  This implements a "switch-on-type" approach to multiple dispatch,
 *  and it can handle any number of dynamically-deduced arguments. The
 *  type of each argument is determined in order, first to last, by
 *  checking a user-provided list of possible dynamic types. The
 *  checks are done by use of `dynamic_cast`, so there is extensive
 *  use of Runtime Type Information (RTTI). If an argument's dynamic
 *  type cannot be deduced, it is left to the user to handle dispatch
 *  errors. How that is handled is entirely outside the scope of this
 *  dispatcher; for more information, see @ref
 *  switch-dispatch-usage-functor "the expections on functors".
 *
 *  @subsection switch-dispatch-algo-inspiration Inspiration
 *
 *  This is inspired by the StaticDispatcher in _Modern C++ Design_ by
 *  Alexei Alexandrescu, with some improvements for modern C++
 *  standards. Most notably, this seamlessly handles an arbitrary
 *  number of arguments, whereas that reference only demonstrates
 *  double dispatch, the two-argument case. This also admits
 *  additional "unclosed" arguments (i.e., not held as members in the
 *  functor), though this is somewhat clunky and not strictly
 *  necessary (because they could just be closed in the functor).
 *
 *  @section switch-dispatch-usage Usage
 *
 *  Multiple dispatch should always be hidden in at least one layer of
 *  indirection and should not be part of a public implementation
 *  (i.e., "client code"). There are two components to using this
 *  dispatcher, preparation of the functor and the multiple dispatch
 *  call site. These are covered in more detail below.
 *
 *  @subsection switch-dispatch-usage-functor Functor Preparation
 *
 *  This section details the requirements on the functor that is
 *  passed into the dispatcher.
 *
 *  The dispatcher is responsible for determining the dynamic type of
 *  each "virtual" argument; there is no way for it to dispatch
 *  directly to an overloaded function (since function names are not
 *  first-class symbols as they are in, say, LISP languages). Thus we
 *  take the standard approach of adding a layer of indirection,
 *  namely running dispatch through an object with suitably overloaded
 *  member functions. This object is a "functor" (Alexandrescu calls
 *  them "executors"), a callable object.
 *
 *  The functor is required to have `operator()` implemented for every
 *  combination of types that is dispatchable. For dispatch to have
 *  guaranteed success, the overload set must contain every possible
 *  combination of types from the given typelists, and every possible
 *  dynamic type for each argument must be present in the given
 *  typelists. Additionally, each overload must have the same return
 *  type. Note that templates or "partially dynamically-typed"
 *  overloads are able to cover various cases, as needed. For example,
 *  if (some of) the overload set is already available as free
 *  functions, a template would be an easy way to thunk the dispatch
 *  to these free functions.
 *
 *  While it is *strongly* encouraged to treat the functor as a
 *  closure around the non-deduced arguments, it is possible to expose
 *  additional "unenclosed" arguments that are not deduced in the
 *  functor interface. These arguments must be positioned *before* the
 *  deduced arguments in formal argument list for `operator()`. For
 *  example, the following is a valid use of an additional argument:
 *
 *  @code{.cpp}
 *  struct MyFunctor {
 *    void operator=()(int x, deduced& a, deduced& b) {...}
 *  };
 *  @endcode
 *
 *  The following is an *invalid* use of an additional argument:
 *
 *  @code{.cpp}
 *  struct MyFunctor {
 *    // ERROR: Additional argument splits deduced arguments
 *    void operator=()(deduced& a, int x, deduced& b) {...}
 *    // ERROR: Additional argument follows deduced arguments
 *    void operator=()(deduced& a, deduced& b, int x) {...}
 *  };
 *  @endcode
 *
 *  The reason for this restriction is technical, and may be lifted in
 *  the future. Note that the ordering of formal arguments to the
 *  dispatcher will be given in a @ref
 *  switch-dispatch-usage-call-site-arguments "different order".
 *
 *  @subsubsection switch-dispatch-usage-functor-errors Error handling
 *
 *  Handling errors is deferred to the functor as well. There are two
 *  types of possible errors that can come out of the dynamic dispatch
 *  process, and the functor class must provide a mechanism for
 *  dealing with each of them.
 *
 *  First, the dynamic type of an argument might not be found in that
 *  argument's typelist. For this, the functor is required to provide
 *  the function `ReturnT DeductionError(...)`. Currently, the
 *  argument list must be variadic; this is a detail of the dispatch
 *  engine that is being ironed out and will hopefully disappear. When
 *  that happens, the requirement will be "... the function `ReturnT
 *  DeductionError(base_typed_signature)`.
 *
 *  Second, the functor may not be callable with the deduced
 *  types. The functor is required to provide a function equivalent to
 *  `ReturnT DispatchError(Args)` in this case, where `Args` matches
 *  the argument list for `operator()` with dynamically-deduced
 *  arguments replaced by their respective base-class references. More
 *  complex techniques (such as templates) could also be used to
 *  provide more detailed functionality.
 *
 *  Ultimately, what happens inside these error-handling functions is
 *  up to the implementation of the functor; no expection or
 *  requirement is imposed by this dispatcher. That is, these cases
 *  are only known to be errors with respect to the dynamic dispatch
 *  engine; it is use-case-specific whether this constitutes a program
 *  error. These functions merely provide a signal to the functor that
 *  this has situation has occurred.
 *
 *  It is important to note that these functions are always required
 *  to be present in a functor. There may be particular use-cases of
 *  this dispatcher that can be implemented such that these cases
 *  cannot occur at runtime; the error functions are still required to
 *  be present. They may be empty.
 *
 *  @subsection switch-dispatch-usage-call-site Call-site Particulars
 *
 *  This section details the use-patterns and idiosyncracies of using
 *  this dispatcher to achieve multiple dispatch.
 *
 *  It bears repeating that this dispatch engine does not directly
 *  operate on overloaded functions; it requires @ref
 *  switch-dispatch-usage-functor "functors with special structure".
 *  Once that has been designed as described, usage is
 *  straight-forward. First, the template arguments to the dispatcher
 *  must be created. Then, the arguments to the dispatcher must be
 *  ordered correctly.
 *
 *  @subsubsection switch-dispatch-usage-call-site-tparams Template Parameters
 *
 *  For a functor with `N` dynamically-deduced arguments, there will
 *  be `2+2*N` template parameters to the dispatcher. The first two
 *  are very simple: the type of the functor and the type that is
 *  returned by its `operator=()` (or the overload set that will be
 *  exploited in this dispatch). Following that, the remaining `2*N`
 *  arguments must be given in pairs: first a base type, then a list
 *  of concrete types against which to test the formal argument. These
 *  must be given in the same order as the dynamically-deduced formal
 *  arguments, and there must be one pair for each formal argument,
 *  even if that means repeating pairs. This may be optimized away in
 *  the future.
 *
 *  @subsubsection switch-dispatch-usage-call-site-arguments Formal Arguments
 *
 *  The dispatcher exposes a single static API: `Exec(...)`. This
 *  function has return type as specified in the template
 *  parameters. The arguments are as follows:
 *
 *    -# A functor object, by value.
 *    -# The arguments that will be dynamically deduced, in the same
 *       order that they will be passed to the functor's `operator()`.
 *    -# The extra "unclosed" arguments, in the same order that they will
 *       be passed to the functor's `operator()`.
 *
 *  Note that these last two groups are ordered differently than when
 *  implementing the functor. This is intentional. Work is in-progress
 *  to resolve this confusion.
 *
 *  @warning This method of multiple dispatch is robust, but it relies
 *  on `dynamic_cast` to check the type of each argument. This heavy
 *  use of RTTI could affect performance if not used carefully. It is
 *  left to users of this dispatch engine to determine whether this
 *  cost is acceptable. In general, it is advisable to avoid multiple
 *  dispatch issues inside tight loops and other performance-critical
 *  sections.
 *
 *  @warning If the functor is implemented using templates, this could
 *  implicitly instantiate all combinations of parameters if care has
 *  not been taken to prevent this. If this incurs too high a
 *  compilation cost, perhaps consider controlling instantiation via
 *  explicit template instantiation, using ETI declarations where
 *  appropriate.
 *
 */
template <typename FunctorT, typename ReturnT, typename... ArgumentTs>
class SwitchDispatcher;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename FunctorT,
          typename ReturnT,
          typename ThisBase,
          typename ThisList,
          typename... ArgumentTs>
class SwitchDispatcher<FunctorT, ReturnT, ThisBase, ThisList, ArgumentTs...>
{
    static_assert(sizeof...(ArgumentTs) % 2 == 0,
                  "Must pass ArgumentTs as (Base, TL<DTypes>).");

public:
    template <typename... Args>
    static ReturnT Exec(FunctorT F, ThisBase& arg, Args&&... others)
    {
        using Head = meta::tlist::Car<ThisList>;
        using Tail = meta::tlist::Cdr<ThisList>;

        if (auto* arg_dc = dynamic_cast<Head*>(&arg))
            return SwitchDispatcher<FunctorT, ReturnT, ArgumentTs...>::Exec(
                F, std::forward<Args>(others)..., *arg_dc);
        else
            return SwitchDispatcher<FunctorT,
                                    ReturnT,
                                    ThisBase,
                                    Tail,
                                    ArgumentTs...>::Exec(F,
                                                         arg,
                                                         std::forward<Args>(
                                                             others)...);
    }
};

// Base case
template <typename FunctorT, typename ReturnT>
class SwitchDispatcher<FunctorT, ReturnT>
{
    template <typename... Ts>
    using Invocable = meta::IsInvocableVT<FunctorT, Ts...>;

public:
    template <typename... Args, meta::EnableWhenV<Invocable<Args...>, int> = 0>
    static ReturnT Exec(FunctorT F, Args&&... others)
    {
        return F(std::forward<Args>(others)...);
    }

    // All types were deduced, but there is no suitable dispatch for
    // this case.
    template <typename... Args,
              meta::EnableUnlessV<Invocable<Args...>, int> = 0>
    static ReturnT Exec(FunctorT F, Args&&... args)
    {
        return F.DispatchError(std::forward<Args>(args)...);
    }
};

// Deduction failure case
template <typename FunctorT,
          typename ReturnT,
          typename ThisBase,
          typename... ArgumentTs>
class SwitchDispatcher<FunctorT,
                       ReturnT,
                       ThisBase,
                       meta::tlist::Empty,
                       ArgumentTs...>
{
public:
    template <typename... Args>
    static ReturnT Exec(FunctorT F, Args&&... args)
    {
        return F.DeductionError(std::forward<Args>(args)...);
    }
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace multimethods
} // namespace h2
#endif // H2_PATTERNS_MULTIMETHODS_SWITCHDISPATCHER_HPP_
