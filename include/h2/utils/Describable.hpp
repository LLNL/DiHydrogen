////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <ostream>
#include <sstream>
#include <string>

namespace h2
{

// Should we control the verbosity level at the instance level
// (function parameter), or via a global envvar or the like?
class Describable
{
public:
    Describable() = default;
    virtual ~Describable() = default;

    /** @brief Print a short (one-line) description to the stream.
     *
     *  This can be as simple as a type name or may contain more
     *  information. This should NOT contain newline or other control
     *  characters, though this is not enforced in any way.
     */
    virtual void short_describe(std::ostream& os) const = 0;

    /** @brief Print a full description to the stream.
     *
     *  By default, this calls short_describe. However, a class may
     *  implement this function to include any information deemed
     *  interesting to the developer of the class. It should be
     *  expected that this may contain newline characters, etc.
     */
    virtual void describe(std::ostream& os) const { return short_describe(os); }

    std::string short_description() const
    {
        std::ostringstream os;
        this->short_describe(os);
        return os.str();
    }

    std::string description() const
    {
        std::ostringstream os;
        this->describe(os);
        return os.str();
    }
}; // class Describable

namespace internal
{
struct FullDescription
{
    Describable const* obj;
    FullDescription(Describable const& obj_) : obj{&obj_} {}
};
} // namespace internal

inline internal::FullDescription describe(Describable const& obj)
{
    return internal::FullDescription{obj};
}

} // namespace h2

inline std::ostream& operator<<(std::ostream& os,
                                h2::internal::FullDescription const& desc)
{
    desc.obj->describe(os);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, h2::Describable const& obj)
{
    obj.short_describe(os);
    return os;
}
