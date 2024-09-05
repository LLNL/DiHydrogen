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
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>

namespace h2
{
namespace factory
{
/** @class CopyFactory
 *  @brief A factory that abstractly copies various types.
 *
 *  This is a modification of the CloneFactory design in _Modern C++
 *  Design_ by Alexei Alexandrescu. The updates include using STL
 *  smart pointers and function objects, as well as std::type_index,
 *  which did not exist at the time. Additionally, a few factory
 *  metadata queries are exposed.
 *
 *  The intended use-case here is when users of a class hierarchy
 *  require a unified copy interface that may or may not be available
 *  in the derived classes. _Modern C++ Design_ highlights a few of
 *  these situations.
 *
 *  Like many conveniently-programmed things in C++, this class makes
 *  heavy use of RTTI.
 *
 *  @tparam AbstractType  The base class of the types being constructed.
 *  @tparam BuilderType   The functor type that builds concrete types.
 *  @tparam ErrorPolicy   The policy for handling errors.
 */
template <typename AbstractType,
          typename BuilderType =
              std::function<std::unique_ptr<AbstractType>(AbstractType const&)>,
          template <typename, typename> class ErrorPolicy = DefaultErrorPolicy>
class CopyFactory : private ErrorPolicy<std::type_info const&, AbstractType>
{
public:
    using abstract_type = AbstractType;
    using id_type = std::type_info;
    using key_type = std::type_index;
    using builder_type = BuilderType;
    using map_type = std::unordered_map<key_type, builder_type>;
    using size_type = typename map_type::size_type;

public:
    /** @brief Register a new builder for things of type @c id */
    bool register_builder(id_type const& id, builder_type builder)
    {
        return map_
            .emplace(std::piecewise_construct,
                     std::forward_as_tuple(std::type_index(id)),
                     std::forward_as_tuple(std::move(builder)))
            .second;
    }

    /** @brief Unregister the current builder for things of type @c id. */
    bool unregister(id_type const& id)
    {
        return (map_.erase(std::type_index(id)) == 1);
    }

    /** @brief Construct a new object forwarding extra arguments to
     *  the builder.
     */
    template <typename... Ts>
    std::unique_ptr<AbstractType> copy_object(AbstractType const& other) const
    {
        auto const& id = typeid(other);
        auto it = map_.find(std::type_index(id));
        if (it != map_.end())
            return (it->second)(other);

        return this->handle_unknown_id(id);
    }

    /** @brief Get the names of all concrete products known to the factory. */
    std::list<std::string> registered_types() const
    {
        std::list<std::string> names;
        for (auto const& x : map_)
            names.push_back(x.first.name());

        return names;
    }

    /** @brief Get the number of products known to the factory. */
    size_type size() const noexcept { return map_.size(); }

private:
    map_type map_;
};

} // namespace factory
} // namespace h2
