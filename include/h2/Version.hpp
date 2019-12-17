#ifndef H2_VERSION_HPP_
#define H2_VERSION_HPP_

#include <string>

/** @namespace h2
 *  @brief The main namespace for DiHydrogen.
 */

namespace h2
{
/** @brief Get the version string for DiHydrogen
 *  @returns A string of the format "MAJOR.MINOR.PATCH".
 */
std::string Version() noexcept;

} // namespace h2
#endif // H2_VERSION_HPP_
