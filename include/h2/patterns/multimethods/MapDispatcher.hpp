////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2/meta/Core.hpp"
#include "h2/meta/TypeList.hpp"

#include <functional>
#include <map>
#include <stdexcept>
#include <typeindex>
#include <type_traits>

namespace h2
{
namespace multimethods
{

/** @brief An exception class to throw when a valid dispatch cannot be
 *         found.
 *  @todo Move elsewhere?
 */
class NoDispatchAdded : public std::runtime_error
{
public:
  NoDispatchAdded() : std::runtime_error{"No dispatch entry for given types"} {}
};

/** @brief Caster that uses dynamic_cast for safety.
 *
 *  This can also be required for certain cases, mostly involving
 *  diamonds in the inheritance graph.
 *
 *  @tparam To The type being cast to. This should inherit From.
 *  @tparam From The type being cast from. This should be a base class
 *               of To.
 *
 *  @todo Maybe move elsewhere?
 */
template <typename To, typename From>
struct DynamicDownCaster
{
  static_assert(
    std::is_base_of_v<std::decay_t<From>, std::decay_t<To>>,
    "Can only cast between classes with a polymorphic relationship.");
  using OutT =
    meta::IfThenElse<std::is_const_v<From>, std::add_const_t<To>, To>;
  static OutT& cast(From& b) { return dynamic_cast<OutT&>(b); }
};

/** @brief Caster that uses static_cast for speed.
 *
 *  This can fail in certain cases, in which case dynamic_cast must be
 *  used. The compiler will complain in these cases, and a user can
 *  switch to DynamicDownCaster or write a custom Caster to handle the
 *  corner cases of their situation.
 *
 *  @tparam To The type being cast to. This should inherit From.
 *  @tparam From The type being cast from. This should be a base class
 *               of To.
 */
template <typename To, typename From>
struct StaticDownCaster
{
  static_assert(
    std::is_base_of_v<std::decay_t<From>, std::decay_t<To>>,
    "Can only cast between classes with a polymorphic relationship.");
  using OutT =
    meta::IfThenElse<std::is_const_v<From>, std::add_const_t<To>, To>;
  static OutT& cast(From& b) { return static_cast<OutT&>(b); }
};

template <typename ReturnT,
          typename BaseArgsTL,
          template <class, class> class CasterT = DynamicDownCaster>
class MapDispatcher;

/** @brief Log-time multimethod
 *
 *  Strongly inspired by Chapter 11 of _Modern C++ Design_ by Andrei
 *  Alexandrescu, but generalized to arbitrary (n-ary) dispatch and
 *  updated to actually modern C++ by TRB.
 *
 *  @tparam ReturnT  The type returned by the functions being
 *                   registered. This obviously must be the same for
 *                   every functor registered in the dispatcher.
 *  @tparam BaseArgs The base class types used to generate the
 *                   polymorphic interface. These should be
 *                   cv-quailified, but NOT references (the reference
 *                   are added automatically by this class template).
 *  @tparam CasterT  The casting policy to use to convert base-class
 *                   arguments to their concrete type.
 *
 *  @todo Treat inheritance properly?
 *  @todo Allow extra arguments?
 */
template <typename ReturnT,
          typename... BaseArgs,
          template <class, class> class CasterT>
class MapDispatcher<ReturnT, meta::TL<BaseArgs...>, CasterT>
{
  /** @brief The type of the key used in the dispatch map. */
  using KeyT = meta::tlist::ToTuple<
    meta::tlist::Repeat<std::type_index, sizeof...(BaseArgs)>>;

  /** @brief The type of function being held. */
  using FunctionT = std::function<ReturnT(BaseArgs&...)>;

  /** @brief Mapping from type_indices to functors */
  using MapT = std::map<KeyT, FunctionT>;

  /** @brief The "dispatch map" storage.
   *  @todo Replace with faster lookup data structure (e.g., a sorted
   *        vector using std::lower_bound for search)
   */
  MapT m_dispatch;

public:
  /** @brief Register a new entry in the dispatch map.
   *
   *  @tparam Ts (User-provided)The (concrete) types for which to
   *             register this entry.
   *  @tparam F (Inferred) The type of the functor being registered.
   *
   *  @param[in] f The functor to register for the types given in Ts.
   */
  template <typename... Ts, typename F>
  void add(F f)
  {
    static_assert(sizeof...(Ts) == sizeof...(BaseArgs),
                  "Number of arguments must match.");
    KeyT const key = {std::type_index(typeid(Ts))...};
    m_dispatch[key] = [f = std::move(f)](BaseArgs&... args) mutable {
      return f(CasterT<Ts, BaseArgs>::cast(args)...);
    };
  }

  /** @brief Register an explicit function pointer in the dispatch
   *         map.
   *
   *  This version of add() does a fun thing where the compiler can
   *  bind the address of a function directly if it is passed as a
   *  nontype template parameter, potentially saving an indirect call.
   *  The annoying bit is that it seems impossible to infer the
   *  function type explicitly (that is, as R(*)(Args...)). Thus one
   *  must either name the function pointer type directly at the call
   *  site, or simply pass `add<decltype(f), f,...>()`.
   *
   *  @tparam Fn (User-provided) The type of the function being
   *             registered.
   *  @tparam f  (User-provided) The address of the function being
   *             registered.
   *  @tparam Ts (User-provided)The (concrete) types for which to
   *             register this entry.
   */
  template <typename Fn, Fn f, typename... Ts>
  void add()
  {
    static_assert(sizeof...(Ts) == sizeof...(BaseArgs),
                  "Number of arguments must match.");
    KeyT const key = {std::type_index(typeid(Ts))...};
    m_dispatch[key] = [](BaseArgs&... args) mutable {
      return f(CasterT<Ts, BaseArgs>::cast(args)...);
    };
  }

  /** @brief Call the held function using the given arguments.
   *
   *  @param args The arguments on which to dispatch.
   *
   *  @throws NoDispatchAdded if the combination of dynamic types
   *          doesn't have an entry in the dispatch map.
   */
  ReturnT call(BaseArgs&... args)
  {
    KeyT const key = {std::type_index(typeid(args))...};
    auto iter = m_dispatch.find(key);
    if (iter == cend(m_dispatch))
      throw NoDispatchAdded{};
    return (iter->second)(args...);
  }

};  // class MapDispatcher

/** @brief A helper macro for saving some typing when registering any
 *         general functor.
 */
#define H2_MDISP_ADD(dispatcher, fn, ...)                                      \
  dispatcher.template add<__VA_ARGS__>(fn)

/** @brief A helper macro for saving some typing when registering a
 *         raw function pointer.
 */
#define H2_MDISP_ADD_FP(dispatcher, fn, ...)                                   \
  dispatcher.template add<decltype(&fn), &fn, __VA_ARGS__>()

}  // namespace multimethods
}  // namespace h2
