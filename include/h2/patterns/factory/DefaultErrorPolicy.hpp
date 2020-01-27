#ifndef H2_PATTERNS_FACTORY_DEFAULTERRORPOLICY_HPP_
#define H2_PATTERNS_FACTORY_DEFAULTERRORPOLICY_HPP_

#include <memory>
#include <stdexcept>

namespace h2
{
namespace factory
{

/** \class DefaultErrorPolicy
 *  \brief Handle unknown keys by throwing exceptions.
 */
template <typename IdType, class ObjectType>
struct DefaultErrorPolicy
{
    struct UnknownIDError : public std::exception
    {
        const char* what() const noexcept override
        {
            return "Unknown type identifier.";
        }
    };

    std::unique_ptr<ObjectType> handle_unknown_id(IdType const&) const
    {
        throw UnknownIDError();
    }
};// struct DefaultErrorPolicy

} // namespace factory
} // namaspace h2
#endif // H2_PATTERNS_FACTORY_DEFAULTERRORPOLICY_HPP_
