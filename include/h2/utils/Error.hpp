#ifndef H2_UTILS_ERROR_HPP_
#define H2_UTILS_ERROR_HPP_

#include <string>

/** @file Error.hpp
 *
 *  A collection of macros and other simple constructs for reporting
 *  and handling errors.
 */

/** @def H2_ADD_FORWARDING_EXCEPTION(name, parent)
 *  @brief Define a class that forwards all arguments to its parent.
 *
 *  This is particularly useful for creating inherited exceptions
 *  from, for example, `std::runtime_error`.
 *
 *  @param name The name of the new class.
 *  @param parent The name of the parent class.
 */
#define H2_DEFINE_FORWARDING_EXCEPTION(name, parent)           \
    class name : public parent                                 \
    {                                                          \
    public:                                                    \
        /* @brief Constructor */                               \
        template <typename... Ts>                              \
        name(Ts&&... args) : parent(std::forward<Ts>(args)...) \
        {}                                                     \
    }

/** @def H2_ASSERT(cond, excptn, msg)
 *  @brief Check that the condition is true and throw an exception if
 *         not.
 *
 *  @param cond The condition to test. Must be a boolean value.
 *  @param excptn The exception to throw if `cond` evaluates to
 *                `false`.
 *  @param ... The arguments to pass to the exception.
 */
#define H2_ASSERT_MSG(cond, excptn, ...)        \
    if (!(cond))                                \
        throw excptn(__VA_ARGS__);

namespace h2
{

/** @brief A function to break on when debugging.
 *
 *  @param[in] msg A value that should be available when breaking.
 */
void break_on_me(std::string const& msg = "");

} // namespace h2
#endif // H2_UTILS_ERROR_HPP_
