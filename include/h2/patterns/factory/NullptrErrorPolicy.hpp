#ifndef H2_PATTERNS_FACTORY_NULLPTRERRORPOLICY_HPP_
#define H2_PATTERNS_FACTORY_NULLPTRERRORPOLICY_HPP_

#include <memory>

namespace h2
{
namespace factory
{

/** \class NullptrErrorPolicy
 *  \brief Returns a nullptr when given an unknown key.
 */
template <typename IdType, class ObjectType>
struct NullptrErrorPolicy
{
    std::unique_ptr<ObjectType> handle_unknown_id(IdType const&) const
    {
        return nullptr;
    }
};// struct NullptrErrorPolicy

} // namespace factory
} // namaspace h2
#endif // H2_PATTERNS_FACTORY_NULLPTRERRORPOLICY_HPP_
