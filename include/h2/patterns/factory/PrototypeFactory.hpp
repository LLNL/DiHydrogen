////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_PATTERNS_FACTORY_PROTOTYPEFACTORY_HPP_
#define H2_PATTERNS_FACTORY_PROTOTYPEFACTORY_HPP_

#include "DefaultErrorPolicy.hpp"

#include <functional>
#include <list>
#include <memory>
#include <unordered_map>
#include <utility>

namespace h2
{
namespace factory
{
/** @class PrototypeFactory
 *  @brief Factory that returns copies of prototypes.
 *
 *  This class is designed to be a "prototype repository", of the sort
 *  described in _Design Patterns: Elements of Reusable
 *  Object-Oriented Software_ by Erich Gamma et al. The design is
 *  inspired by the factory designs in _Modern C++ Designs_ by Alexei
 *  Alexandrescu, though there is no such class described in that
 *  book.
 *
 *  The interesting component of this class is the CopyPolicy, which
 *  describes how to produce new concrete products from the held
 *  prototypical instance. The signature that this policy is required
 *  to provide is:
 *
 *  @code{.cpp}
 *     std::unique_ptr<AbstractType> Copy(AbstractType const&, ...) const;
 *  @endcode
 *
 *  The variadic parameter is optional but supported in the factory to
 *  enable greater flexibility in cloning.
 *
 *  The primary restriction is that every prototype must use the same
 *  CopyPolicy (which is why it's a policy). For more flexible
 *  copying, perhaps consider using CopyFactory with closures around
 *  prototypes as the builders.
 *
 *  @tparam AbstractType  The base class of the types being constructed.
 *  @tparam IdType        The index type used to differentiate concrete types.
 *  @tparam CopyPolicy    A policy that describes how each prototype is copied.
 *  @tparam ErrorPolicy   The policy for handling errors.
 */
template <
    typename AbstractType,
    typename IdType,
    typename CopyPolicy,
    template <typename, typename> class ErrorPolicy = DefaultErrorPolicy>
class PrototypeFactory : private CopyPolicy,
                         private ErrorPolicy<IdType, AbstractType>
{
public:
    using abstract_type = AbstractType;
    using id_type = IdType;
    using abstract_ptr_type = std::unique_ptr<abstract_type>;
    using map_type = std::unordered_map<id_type, abstract_ptr_type>;
    using size_type = typename map_type::size_type;

public:
    /** @brief Register a new prototype for things of type @c id. */
    bool register_prototype(
        id_type const& id, std::unique_ptr<abstract_type>&& prototype)
    {
        return map_
            .emplace(
                std::piecewise_construct, std::forward_as_tuple(id),
                std::forward_as_tuple(std::move(prototype)))
            .second;
    }

    /** @brief Register a new prototype for things of type @c id. */
    bool
    register_prototype(id_type&& id, std::unique_ptr<abstract_type>&& prototype)
    {
        return map_
            .emplace(
                std::piecewise_construct, std::forward_as_tuple(std::move(id)),
                std::forward_as_tuple(std::move(prototype)))
            .second;
    }

    /** @brief Unregister the current prototype for things of type @c id.
     *  @note This will free the underlying prototype instance.
     */
    bool unregister(id_type const& id) { return (map_.erase(id) == 1); }

    /** @brief Construct a new object forwarding extra arguments to
     *  the copy policy.
     */
    template <typename... Ts>
    std::unique_ptr<AbstractType>
    copy_prototype(IdType const& id, Ts&&... Args) const
    {
        auto it = map_.find(id);
        if (it != map_.end())
            return this->Copy(*(it->second), std::forward<Ts>(Args)...);

        return this->handle_unknown_id(id);
    }

    /** @brief Get the names of all prototypes known to the factory. */
    std::list<id_type> registered_keys() const
    {
        std::list<id_type> names;
        for (auto const& x : map_)
            names.push_back(x.first);

        return names;
    }

    /** @brief Get the number of builders known to the factory. */
    size_type size() const noexcept { return map_.size(); }

private:
    map_type map_;
}; // class PrototypeFactory

} // namespace factory
} // namespace h2
#endif /* H2_PATTERNS_FACTORY_PROTOTYPEFACTORY_HPP_ */
