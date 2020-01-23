// @H2_LICENSE_TEXT@

#ifndef H2_META_TYPELIST_UNIQUE_HPP_
#define H2_META_TYPELIST_UNIQUE_HPP_

#include "TypeList.hpp"
#include "LispAccessors.hpp"
#include "Remove.hpp"

#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{

/** @brief Create a copy of the list with duplicates removed. */
template <typename List>
struct UniqueT;

/** @brief Create a copy of the list with duplicates removed. */
template <typename List>
using Unique = Force<UniqueT<List>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Base case
template <>
struct UniqueT<Empty>
{
    using type = Empty;
};

// Recursive case
template <typename List>
struct UniqueT
{
private:
    using Head_ = Car<List>;
    using Tail_ = Cdr<List>;
    using UniqueTail_ = Unique<Tail_>;
public:
    using type = Cons<Head_, Remove<UniqueTail_, Head_>>;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
}// namespace tlist
}// namespace meta
}// namespace h2
#endif // H2_META_TYPELIST_UNIQUE_HPP_
