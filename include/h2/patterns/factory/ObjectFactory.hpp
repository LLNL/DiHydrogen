////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

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
/** @class ObjectFactory
 *  @brief Generic factory template.
 *
 *  @tparam AbstractType  The base class of the types being constructed.
 *  @tparam IdType        The index type used to differentiate concrete types.
 *  @tparam BuilderType   The functor type that builds concrete types.
 *  @tparam ErrorPolicy   The policy for handling errors.
 */
template <typename AbstractType,
          typename IdType,
          typename BuilderType = std::function<std::unique_ptr<AbstractType>()>,
          template <typename, typename> class ErrorPolicy = DefaultErrorPolicy>
class ObjectFactory : private ErrorPolicy<IdType, AbstractType>
{
public:
    using abstract_type = AbstractType;
    using id_type = IdType;
    using builder_type = BuilderType;
    using map_type = std::unordered_map<id_type, builder_type>;
    using size_type = typename map_type::size_type;

public:
    /** @brief Register a new builder for things of type @c id */
    bool register_builder(id_type id, builder_type builder)
    {
        return map_
            .emplace(std::piecewise_construct,
                     std::forward_as_tuple(std::move(id)),
                     std::forward_as_tuple(std::move(builder)))
            .second;
    }

    /** @brief Unregister the current builder for things of type @c id. */
    bool unregister(id_type const& id) { return (map_.erase(id) == 1); }

    /** @brief Construct a new object forwarding extra arguments to
     *  the builder.
     */
    template <typename... Ts>
    std::unique_ptr<AbstractType> create_object(IdType const& id,
                                                Ts&&... Args) const
    {
        auto it = map_.find(id);
        if (it != map_.end())
            return (it->second)(std::forward<Ts>(Args)...);

        return this->handle_unknown_id(id);
    }

    /** @brief Get the names of all builders known to the factory. */
    std::list<id_type> registered_ids() const
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
};

} // namespace factory
} // namespace h2
