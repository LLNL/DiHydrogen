////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <h2/utils/Error.hpp>

namespace h2
{
void break_on_me(std::string const& msg)
{
    char const volatile* x = msg.data();
    (void) x;
}
} // namespace h2
